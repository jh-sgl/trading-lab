import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from util.registry import register_dataset

from .const import Key, Num


@register_dataset("ExtremaDetectorRevis")
class ExtremaDetectorRevisDataset(Dataset):
    def __init__(
        self,
        data_fp: str,
        date_range: tuple[str, str],  # in "HHHH-MM-DD" format
        resample_rule: str,
        input_columns_info: list[str],
        lookback_num: int,
        lookahead_num: int,
        minima_column: str,
        maxima_column: str,
        trade_stop_hour: int,
    ) -> None:
        self._input_columns_info = input_columns_info
        self._lookback_num = lookback_num
        self._lookahead_num = lookahead_num
        self._df = self._load_data(data_fp, date_range, resample_rule)
        self._df, self._dataloader_idx = self._create_columns_for_trade(
            self._df, lookback_num, lookahead_num, minima_column, maxima_column, trade_stop_hour
        )

    def _create_next_opposite_extrema_price(
        self, df: pd.DataFrame, minima_column: str, maxima_column: str
    ) -> pd.Series:
        result_series = pd.Series([np.nan] * len(df), index=df.index)

        idx = np.arange(len(df))
        minima_idx = idx[df[minima_column]]
        maxima_idx = idx[df[maxima_column]]

        def _find_next_extrema(source_idx: np.ndarray, target_idx: np.ndarray) -> np.ndarray:
            result = np.full_like(source_idx, fill_value=-1)
            for i, s_idx in enumerate(source_idx):
                future_targets = target_idx[target_idx > s_idx]
                if len(future_targets) > 0:
                    result[i] = future_targets[0]
            return result

        minima_next_maxima = _find_next_extrema(minima_idx, maxima_idx)
        maxima_next_minima = _find_next_extrema(maxima_idx, minima_idx)

        last_price = df[Key.FUTURE_PRICE_CLOSE].iloc[-1]
        if len(minima_next_maxima) > 0:
            for src, tgt in zip(minima_idx, minima_next_maxima):
                result_series.iloc[src] = df.iloc[tgt][Key.FUTURE_PRICE_CLOSE] if tgt != -1 else last_price
        if len(maxima_next_minima) > 0:
            for src, tgt in zip(maxima_idx, maxima_next_minima):
                result_series.iloc[src] = df.iloc[tgt][Key.FUTURE_PRICE_CLOSE] if tgt != -1 else last_price

        return result_series

    def _create_columns_for_trade(
        self,
        df: pd.DataFrame,
        lookback_num: int,
        lookahead_num: int,
        minima_column: str,
        maxima_column: str,
        trade_stop_hour: int,
    ) -> tuple[pd.DataFrame, list[int]]:

        price_group = df[Key.FUTURE_PRICE_CLOSE].groupby(df.index.date)
        df[Key.PRICE_ENTER] = price_group.transform(lambda x: x.shift(-1, fill_value=x.iloc[-1]))
        df[Key.PRICE_EXIT] = (
            df.groupby(df.index.date)
            .apply(lambda x: self._create_next_opposite_extrema_price(x, minima_column, maxima_column))
            .reset_index(level=1)
            .set_index("time")
        )
        df[Key.PRICE_MOVE] = df[Key.PRICE_EXIT] - df[Key.PRICE_ENTER]

        df[Key.PRICE_MOVE_CLIPPED] = df[Key.PRICE_MOVE]
        commission = (df[Key.PRICE_ENTER] + df[Key.PRICE_MOVE]) * Num.COMMISSION_RATE
        is_profitable = (abs(df[Key.PRICE_MOVE]) - Num.SLIPPAGE_PER_EXECUTION * 2 - commission) > 0
        df.loc[~is_profitable, Key.PRICE_MOVE_CLIPPED] = 0

        df[Key.VOLATILITY_50] = df[Key.PRICE_ENTER].diff().abs().rolling(50, min_periods=1).mean()

        df = df[df.index.hour < trade_stop_hour]
        # df = df.dropna(subset=[Key.VOLATILITY_50]).copy()
        # dataloader_idx = df.index.get_indexer(df[df[Key.PRICE_MOVE].notna()].index)
        # dataloader_idx = [idx for idx in dataloader_idx if idx >= lookback_num]
        df = df.dropna(subset=[Key.PRICE_MOVE, Key.VOLATILITY_50]).copy()
        dataloader_idx = [idx for idx in range(len(df)) if idx >= lookback_num]
        return df, dataloader_idx

    @property
    def _columns_to_use(self) -> list[str]:
        return self._input_columns_info.keys()

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

        inputs = past_rows[self._input_columns_info.keys()].copy()
        for col_name, normalize in self._input_columns_info.items():
            if normalize == "first_row":
                inputs[col_name] -= inputs[col_name].iloc[0]
            elif normalize == "standardized_normal":
                inputs[col_name] = (inputs[col_name] - inputs[col_name].mean()) / (inputs[col_name].std() + Num.EPS)
            elif normalize == None:
                pass
            else:
                raise ValueError(f"Unknown normalization mode: {normalize}")

        inputs = torch.tensor(inputs.values.astype(float), dtype=torch.float)
        label = torch.tensor(current_row[Key.PRICE_MOVE_CLIPPED], dtype=torch.float)
        timestamp = torch.tensor(current_row.name.value, dtype=torch.int64)
        return inputs, label, timestamp

    def __len__(self) -> int:
        return len(self._dataloader_idx)
