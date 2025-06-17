import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from util.registry import register_dataset

from .const import Key, Num


@register_dataset("STReLUpstream")
class STReLUpstreamDataset(Dataset):
    def __init__(
        self,
        data_fp: str,
        date_range: tuple[str, str],  # in "HHHH-MM-DD" format
        resample_rule: str,
        input_columns: list[str],
        lookback_num: int,
        lookahead_num: int,
    ) -> None:
        self._input_columns = input_columns
        self._lookback_num = lookback_num
        self._lookahead_num = lookahead_num
        self._df, self._dataloader_idx = self._load_data(
            data_fp, date_range, resample_rule, lookback_num, lookahead_num
        )
        self._price_seasonal, self._price_trend = self._decompose_seasonal_trend(
            self._df[Key.FUTURE_PRICE_CLOSE].groupby(self._df.index.date), lookback_num
        )

    def _decompose_seasonal_trend(self, group: pd.Series, kernel_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = []
        lengths = []
        for _, g in group:
            seq = torch.from_numpy(g.values).float().cuda().unsqueeze(-1)
            x.append(seq)
            lengths.append(seq.shape[0])

        x = pad_sequence(x, batch_first=True)

        front_pad = kernel_size // 2
        end_pad = kernel_size - front_pad - 1
        front = x[:, :1, :].expand(-1, front_pad, -1)
        end = x[:, -1:, :].expand(-1, end_pad, -1)
        x_padded = torch.cat([front, x, end], dim=1)

        moving_avg = F.avg_pool1d(x_padded.permute(0, 2, 1), kernel_size=kernel_size, stride=1)
        moving_avg = moving_avg.permute(0, 2, 1)
        residue = x - moving_avg

        moving_avg = torch.cat([moving_avg[i, : lengths[i], 0] for i in range(len(lengths))])
        residue = torch.cat([residue[i, : lengths[i], 0] for i in range(len(lengths))])
        return moving_avg.cpu(), residue.cpu()

    @property
    def _columns_to_use(self) -> list[str]:
        return self._input_columns

    def _filter_out_nan(self, data: pd.DataFrame, columns_to_use: list[str]) -> pd.DataFrame:
        return data.dropna(subset=set(columns_to_use)).reset_index().set_index("time")

    def _load_data(
        self,
        data_fp: str | Path,
        date_range: tuple[str, str],
        resample_rule: str,
        input_lookback_num: int,
        label_lookahead_num: int,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        orig_data = pd.read_parquet(data_fp)

        data = self._filter_out_nan(orig_data, self._columns_to_use)
        data = data[data.resample_rule == resample_rule].copy()
        time = data.index

        start, end = date_range
        logging.info(f"Using data from ({start}) to ({end})")

        data = data[(start <= time) & (time <= end)]

        def _get_dataloader_idx(group, input_lookback_num, label_lookahead_num):
            start = input_lookback_num
            end = len(group) - label_lookahead_num
            if end <= start:
                return pd.DatetimeIndex([])
            return group.iloc[start:end].index

        dataloader_idx = data.groupby(data.index.date).apply(
            lambda x: _get_dataloader_idx(x, input_lookback_num, label_lookahead_num)
        )
        dataloader_idx = pd.DatetimeIndex([ts for idx in dataloader_idx for ts in idx])
        dataloader_idx = data.index.get_indexer(dataloader_idx)
        return data, dataloader_idx

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[Key, torch.Tensor], dict[Key, torch.Tensor]]:
        current_idx = self._dataloader_idx[idx]

        past_start = current_idx - self._lookback_num
        future_end = current_idx + self._lookahead_num + 1

        past_rows = self._df.iloc[past_start : current_idx + 1]  # current-inclusive
        future_rows = self._df.iloc[current_idx:future_end]  # current-inclusive

        past_inputs = self._prepare_inputs(past_rows, discard_anchor_elem=True)
        past_labels = self._prepare_labels(past_rows, discard_anchor_elem=True)
        future_labels = self._prepare_labels(future_rows, discard_anchor_elem=True)

        return past_inputs, past_labels, future_labels

    def _prepare_inputs(self, past_rows: pd.DataFrame, discard_anchor_elem: bool) -> torch.Tensor:
        inputs = torch.tensor(past_rows[self._input_columns].values, dtype=torch.float32)
        if discard_anchor_elem:
            inputs = inputs[1:]
        return inputs

    def _prepare_labels(self, rows: pd.DataFrame, discard_anchor_elem: bool) -> dict[Key, torch.Tensor]:
        """
        # TODO
        - a set of trend/seasonal version label
        - a set of velocity/accelerate version label
        """
        price = rows[Key.FUTURE_PRICE_CLOSE]

        price_delta = price.diff().fillna(0.0)  # regression: (T)
        sharpe, volatility = self._calc_sharpe_volatility(price_delta)  # regression: (T), (T)
        volume_delta = rows[Key.FUTURE_VOLUME].diff().fillna(0.0)  # regression: (T)

        profit = price_delta.cumsum()  # regression: (T)
        is_profitable = (
            abs(profit) - Num.SLIPPAGE_PER_EXECUTION * 2 - (price + price.iloc[0]) * Num.COMMISSION_RATE
        ) > 0
        profit[~is_profitable] = 0

        minmax_profit = pd.Series(0, index=profit.index)  # classification: (T)
        minmax_profit[profit == profit.max()] = 1
        minmax_profit[profit == profit.min()] = -1

        idx = self._df.index.get_indexer(rows.index)
        price_trend = self._price_trend[idx]  # reconstruction: (T)
        price_seasonal = self._price_seasonal[idx]  # reconstruction: (T)

        ohlc_rank = self._decide_ohlc_rank(
            rows[
                [
                    Key.FUTURE_PRICE_OPEN,
                    Key.FUTURE_PRICE_HIGH,
                    Key.FUTURE_PRICE_LOW,
                    Key.FUTURE_PRICE_CLOSE,
                ]
            ]
        )  # pairwise-ranking: (T, 4)

        def _create_tensor_labels(items):
            items_to_stack = []
            for item in items:
                if isinstance(item, pd.Series):
                    item = torch.from_numpy(item.values)
                elif isinstance(item, np.ndarray):
                    item = torch.from_numpy(item)
                elif isinstance(item, torch.Tensor):
                    pass
                else:
                    raise TypeError(f"Not supported type: {type(item)}")
                items_to_stack.append(item)

            # discard first item (anchor point for calculation, as such 0 for most items)
            label = torch.stack(items_to_stack, dim=-1).squeeze(-1).float()
            if discard_anchor_elem:
                label = label[1:]
            return label

        labels = {
            Key.REGRESSION: _create_tensor_labels([price_delta, sharpe, volatility, volume_delta, profit]),
            Key.CLASSIFICATION: _create_tensor_labels([minmax_profit]),
            Key.RECONSTRUCTION: _create_tensor_labels([price_trend, price_seasonal]),
            Key.RANKING: _create_tensor_labels([ohlc_rank]),  # TODO: modify candle ordering -> wick / body
        }
        return labels

    def _decide_ohlc_rank(self, ohlc_df: pd.DataFrame) -> np.ndarray:
        ohlc = ohlc_df.values
        rank = np.argsort(np.argsort(-ohlc, axis=1), axis=1) + 1
        return rank

    def _calc_sharpe_volatility(self, delta: pd.Series) -> tuple[pd.Series, pd.Series]:
        annual = np.sqrt(252)

        returns_mean = delta.expanding().mean()
        returns_std = delta.expanding().std(ddof=0)

        sharpe = returns_mean / (returns_std + Num.EPS) * annual
        volatility = returns_std * annual
        return sharpe, volatility

    def __len__(self) -> int:
        return len(self._dataloader_idx)


@register_dataset("STReLDownstream")
class STReLDownstreamDataset(Dataset):
    def __init__(
        self,
        data_fp: str,
        date_range: tuple[str, str],  # in "HHHH-MM-DD" format
        resample_rule: str,
        input_columns: list[str],
        lookback_num: int,
        lookahead_num: int,
    ) -> None:
        self._input_columns = input_columns
        self._lookback_num = lookback_num
        self._lookahead_num = lookahead_num
        self._df = self._load_data(data_fp, date_range, resample_rule)
        self._df, self._dataloader_idx = self._create_columns_for_trade(self._df, lookback_num, lookahead_num)

    def _create_columns_for_trade(
        self, df: pd.DataFrame, lookback_num: int, lookahead_num: int
    ) -> tuple[pd.DataFrame, list[int]]:
        price_group = df[Key.FUTURE_PRICE_CLOSE].groupby(df.index.date)
        df[Key.PRICE_ENTER] = price_group.transform(lambda x: x.shift(-1, fill_value=x.iloc[-1]))
        df[Key.PRICE_EXIT] = price_group.transform(lambda x: x.shift(-lookahead_num))
        df[Key.PRICE_MOVE] = df[Key.PRICE_EXIT] - df[Key.PRICE_ENTER]

        df[Key.PRICE_MOVE_CLIPPED] = df[Key.PRICE_MOVE]
        commission = (df[Key.PRICE_ENTER] + df[Key.PRICE_MOVE]) * Num.COMMISSION_RATE
        is_profitable = (abs(df[Key.PRICE_MOVE]) - Num.SLIPPAGE_PER_EXECUTION * 2 - commission) > 0
        df.loc[~is_profitable, Key.PRICE_MOVE_CLIPPED] = 0

        df[Key.VOLATILITY_50] = df[Key.PRICE_ENTER].diff().abs().rolling(50, min_periods=1).mean()

        df = df.dropna(subset=[Key.PRICE_MOVE, Key.VOLATILITY_50]).copy()
        dataloader_idx = [idx for idx in range(len(df)) if idx >= lookback_num]
        return df, dataloader_idx

    @property
    def _columns_to_use(self) -> list[str]:
        return list(
            set(
                self._input_columns
                + [
                    Key.FUTURE_PRICE_CLOSE,
                    Key.FUTURE_PRICE_HIGH,
                    Key.FUTURE_PRICE_LOW,
                    Key.FUTURE_PRICE_OPEN,
                    Key.FUTURE_VOLUME,
                ]
            )
        )

    def _filter_out_nan(self, data: pd.DataFrame, columns_to_use: list[str]) -> pd.DataFrame:
        return data.dropna(subset=set(columns_to_use)).reset_index().set_index("time")

    def _load_data(
        self,
        data_fp: str | Path,
        date_range: tuple[str, str],
        resample_rule: str,
    ) -> pd.DataFrame:
        orig_data = pd.read_parquet(data_fp)

        data = self._filter_out_nan(orig_data, self._columns_to_use)
        data = data[data.resample_rule == resample_rule].copy()
        time = data.index

        start, end = date_range
        logging.info(f"Using data from ({start}) to ({end})")

        data = data[(start <= time) & (time <= end)]
        return data

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        current_idx = self._dataloader_idx[idx]

        past_start = current_idx - self._lookback_num + 1  # discard anchor elem (see UpstreamDataset.__getitem__())

        past_rows = self._df.iloc[past_start : current_idx + 1]  # current-inclusive
        current_row = self._df.iloc[current_idx]

        inputs = torch.tensor(past_rows[self._input_columns].values, dtype=torch.float)
        label = torch.tensor(current_row[Key.PRICE_MOVE_CLIPPED], dtype=torch.float)
        timestamp = torch.tensor(current_row.name.value, dtype=torch.int64)
        return inputs, label, timestamp

    def __len__(self) -> int:
        return len(self._dataloader_idx)
