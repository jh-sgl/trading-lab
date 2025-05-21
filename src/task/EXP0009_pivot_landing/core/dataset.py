import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import one_hot
from torch.utils.data import Dataset

from util.const import Num
from util.registry import register_dataset

from .const import DFKey


@register_dataset("basic")
class BasicDataset(Dataset):
    def __init__(
        self,
        data_fp: str,
        date_range: tuple[str, str],  # in "HHHH-MM-DD" format
        resample_rule: str,
        lookback_num: int,
        trade_stop_hour: int,
        add_gumbel_noise: bool,
        repr_fp: str,
        use_repr: bool,
        use_df_inputs: bool,
    ) -> None:
        self._lookback_num = lookback_num
        self._add_gumbel_noise = add_gumbel_noise
        self._df, self._repr, repr_idx, self._repr_for_dbg = self._load_data(
            data_fp, date_range, resample_rule, repr_fp
        )
        self._df, self._dataloader_idx, self._repr_idx = self._create_columns_for_trade(
            self._df, lookback_num, trade_stop_hour, repr_idx
        )
        self._use_repr = use_repr
        self._use_df_inputs = use_df_inputs

    @property
    def label_values(self) -> list[int]:
        return [0, 1, 2, 3, 4, 5, 6, 7]

    def _create_ema_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df[DFKey.PRICE_EMA_5] = df[DFKey.FUTURE_PRICE_CLOSE].ewm(span=5, adjust=False).mean()
        df[DFKey.PRICE_EMA_20] = df[DFKey.FUTURE_PRICE_CLOSE].ewm(span=20, adjust=False).mean()
        df[DFKey.PRICE_EMA_60] = df[DFKey.FUTURE_PRICE_CLOSE].ewm(span=60, adjust=False).mean()
        df[DFKey.PRICE_EMA_120] = df[DFKey.FUTURE_PRICE_CLOSE].ewm(span=120, adjust=False).mean()

        df = df.dropna(subset=[DFKey.PRICE_EMA_5, DFKey.PRICE_EMA_20, DFKey.PRICE_EMA_60, DFKey.PRICE_EMA_120]).copy()
        return df

    def _create_pivot_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        yesterday = df[DFKey.FUTURE_PRICE_CLOSE].copy().groupby(df.index.date).agg(["max", "min", "last"]).shift(1)

        high, low, close = yesterday["max"], yesterday["min"], yesterday["last"]
        pivot = (high + low + close) / 3

        r1 = 2 * pivot - low
        s1 = 2 * pivot - high

        r2 = pivot - low + high
        s2 = pivot + low - high

        r3 = high + 2 * (pivot - low)
        s3 = low - 2 * (high - pivot)

        pivot_df = pd.DataFrame(
            {
                "date": pivot.index,
                DFKey.PIVOT: pivot,
                DFKey.R1: r1,
                DFKey.S1: s1,
                DFKey.R2: r2,
                DFKey.S2: s2,
                DFKey.R3: r3,
                DFKey.S3: s3,
            }
        )

        df = df.merge(pivot_df, left_on="date", right_on="date", how="left").set_index(df.index)

        for source_col, target_col in [
            (DFKey.FUTURE_PRICE_MARKET_CLOSING, DFKey.MARKET_CLOSING_PIVOT_ZONE),
            (DFKey.FUTURE_PRICE_CLOSE, DFKey.PRICE_PIVOT_ZONE),
            (DFKey.PRICE_EMA_5, DFKey.PRICE_EMA_5_PIVOT_ZONE),
            (DFKey.PRICE_EMA_20, DFKey.PRICE_EMA_20_PIVOT_ZONE),
            (DFKey.PRICE_EMA_60, DFKey.PRICE_EMA_60_PIVOT_ZONE),
            (DFKey.PRICE_EMA_120, DFKey.PRICE_EMA_120_PIVOT_ZONE),
        ]:
            price = df[source_col]

            conditions = [
                (price <= df[DFKey.S3]),
                (price > df[DFKey.S3]) & (price <= df[DFKey.S2]),
                (price > df[DFKey.S2]) & (price <= df[DFKey.S1]),
                (price > df[DFKey.S1]) & (price < df[DFKey.PIVOT]),
                (price >= df[DFKey.PIVOT]) & (price < df[DFKey.R1]),
                (price >= df[DFKey.R1]) & (price < df[DFKey.R2]),
                (price >= df[DFKey.R2]) & (price < df[DFKey.R3]),
                (price >= df[DFKey.R3]),
            ]

            df[target_col] = np.select(conditions, self.label_values)
        df = df.dropna(subset=[DFKey.PIVOT]).copy()
        return df

    def _create_columns_for_trade(
        self, df: pd.DataFrame, lookback_num: int, trade_stop_hour: int, repr_idx: pd.Index
    ) -> tuple[pd.DataFrame, list[int]]:
        df = self._create_ema_columns(df)
        df = self._create_pivot_columns(df)

        price_group = df[DFKey.FUTURE_PRICE_OPEN].groupby(df.index.date)
        df[DFKey.PRICE_ENTER] = price_group.transform(lambda x: x.shift(-1, fill_value=x.iloc[-1]))

        df[DFKey.PRICE_EXIT] = df[DFKey.FUTURE_PRICE_MARKET_CLOSING]
        df[DFKey.PRICE_MOVE] = df[DFKey.PRICE_EXIT] - df[DFKey.PRICE_ENTER]

        df[DFKey.PRICE_MOVE_CLIPPED] = df[DFKey.PRICE_MOVE]
        commission = (df[DFKey.PRICE_ENTER] + df[DFKey.PRICE_EXIT]) * Num.COMMISSION_RATE
        is_profitable = (abs(df[DFKey.PRICE_MOVE]) - Num.SLIPPAGE_PER_EXECUTION * 2 - commission) > 0
        df.loc[~is_profitable, DFKey.PRICE_MOVE_CLIPPED] = 0

        df[DFKey.VOLATILITY_50] = df[DFKey.PRICE_ENTER].diff().abs().rolling(50, min_periods=1).mean()

        df = df[df.index.hour < trade_stop_hour]
        df = df.dropna(subset=[DFKey.PRICE_MOVE, DFKey.VOLATILITY_50]).copy()
        dataloader_idx = [idx for idx in range(len(df)) if idx >= lookback_num]
        repr_idx = repr_idx.get_indexer(df.index)
        return df, dataloader_idx, repr_idx

    def _load_data(
        self, data_fp: str | Path, date_range: tuple[str, str], resample_rule: str, repr_fp: str
    ) -> pd.DataFrame:
        orig_data = pd.read_parquet(data_fp)

        data = orig_data[orig_data.resample_rule == resample_rule].copy()
        time = data.index

        start, end = date_range
        logging.info(f"Using data from ({start}) to ({end})")

        data = data[(start <= time) & (time <= end)]

        repr_ = torch.load(repr_fp, weights_only=False)
        repr_idx = pd.to_datetime(repr_["df"]["time"]["open"].dt.floor("5T").values)

        # data = data.loc[repr_idx[repr_idx.isin(data.index)]].copy()
        # data.index.name = "time"

        data = data.dropna(subset=[DFKey.FUTURE_PRICE_CLOSE]).copy()
        return data, repr_["repr"], repr_idx, repr_

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        current_idx = self._dataloader_idx[idx]

        past_start = current_idx - self._lookback_num + 1

        past_rows = self._df.iloc[past_start : current_idx + 1]  # current-inclusive
        current_row = self._df.iloc[current_idx]

        inputs_to_concat = []

        if self._use_df_inputs:
            df_inputs = torch.tensor(
                past_rows[
                    [
                        "future_price_close_disparity_5",
                        "future_price_close_disparity_20",
                        "future_price_close_disparity_100",
                        "future_price_close_disparity_2000",
                        # "future_price_close_maxima_5",
                        # "future_price_close_maxima_20",
                        # "future_price_close_maxima_60",
                        # "future_price_close_maxima_120",
                        # "future_price_close_minima_5",
                        # "future_price_close_minima_20",
                        # "future_price_close_minima_60",
                        # "future_price_close_minima_120",
                        DFKey.PRICE_PIVOT_ZONE,
                        DFKey.PRICE_EMA_5_PIVOT_ZONE,
                        DFKey.PRICE_EMA_20_PIVOT_ZONE,
                        DFKey.PRICE_EMA_60_PIVOT_ZONE,
                        DFKey.PRICE_EMA_120_PIVOT_ZONE,
                    ]
                ].values.astype(float),
                dtype=torch.float,
            )
            df_inputs[:, -5:] = (df_inputs[:, -5:] - 3.5) / 3.5
            inputs_to_concat.append(df_inputs)

            date_token = torch.tensor(pd.factorize(past_rows.index.date)[0], dtype=torch.float).unsqueeze(-1)
            inputs_to_concat.append(date_token)

        if self._use_repr:
            selected_repr_idx = self._repr_idx[past_start : current_idx + 1]
            repr_inputs = self._repr[selected_repr_idx]
            repr_inputs = repr_inputs.reshape(repr_inputs.shape[0], -1)
            inputs_to_concat.append(torch.tensor(repr_inputs, dtype=torch.float32))

        inputs = torch.concat(inputs_to_concat, dim=-1)
        label = self._prepare_label(current_row)

        timestamp = torch.tensor(current_row.name.value, dtype=torch.int64)
        return inputs, label, timestamp

    def _prepare_label(self, current_row: pd.Series) -> torch.Tensor:
        sigma = torch.rand(1) + 0.5
        # sigma = 1.0
        label_gauss = torch.exp(
            -(
                (
                    torch.arange(len(self.label_values), dtype=torch.float32)
                    - current_row[DFKey.MARKET_CLOSING_PIVOT_ZONE]
                )
                ** 2
            )
        ) / (2 * sigma**2)
        label_gauss = label_gauss / label_gauss.sum()

        if self._add_gumbel_noise:
            label_gauss = label_gauss + 0.05 * self._sample_gumbel_noise(label_gauss)
            label_gauss = torch.softmax(label_gauss, dim=0)
        return label_gauss

    def __len__(self) -> int:
        return len(self._dataloader_idx)

    def _sample_gumbel_noise(self, labels: torch.Tensor) -> torch.Tensor:
        U = torch.rand_like(labels)
        return -torch.log(-torch.log(U + Num.EPS) + Num.EPS)


@register_dataset("closingprofit")
class ClosingProfitDataset(BasicDataset):
    def __init__(
        self,
        data_fp: str,
        date_range: tuple[str, str],  # in "HHHH-MM-DD" format
        resample_rule: str,
        lookback_num: int,
        trade_stop_hour: int,
        add_gumbel_noise: bool,
        repr_fp: str,
        use_repr: bool,
        use_df_inputs: bool,
    ) -> None:
        super().__init__(
            data_fp,
            date_range,
            resample_rule,
            lookback_num,
            trade_stop_hour,
            add_gumbel_noise,
            repr_fp,
            use_repr,
            use_df_inputs,
        )

    def _create_soft_label(self, label_val: float, position_threshold: float | None, tau: float = 2.5):
        if position_threshold is None:
            raise ValueError("For soft labels, position_threshold should not be None.")

        pos_label_val = abs(label_val)
        if label_val > position_threshold:
            label = [0, position_threshold, pos_label_val]
        elif label_val < -position_threshold:
            label = [pos_label_val, position_threshold, 0]
        else:
            label = [0, position_threshold, 0]

        label = np.array(label).astype(np.float32)
        exp_x = np.exp((label - np.max(label)) / tau)
        soft_label = exp_x / exp_x.sum(axis=-1, keepdims=True)
        return soft_label

    def _prepare_label(self, current_row: pd.Series) -> torch.Tensor:
        return torch.tensor(self._create_soft_label(current_row[DFKey.PRICE_MOVE], 0.5), dtype=torch.float32)
