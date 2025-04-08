import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from util.registry import register_dataset

from .const import DataKey


@register_dataset("ExtremaDetector")
class ExtremaDetectorDataset(Dataset):
    def __init__(
        self,
        data_fp: str | Path,
        resample_rule: str,
        input_columns: list[str],
        extrema_columns: list[str],
        price_enter_column: str,
        price_exit_column: str,
        date_range: tuple[str, str],  # in "HHHH-MM-DD" format
        input_lookback_num: int,
        predict_lookahead_num: int,
        use_soft_label: bool = False,
        position_threshold: float | None = None,
    ) -> None:

        self._input_columns = input_columns
        self._extrema_columns = extrema_columns
        self._price_enter_column = price_enter_column
        self._price_exit_column = price_exit_column
        self._input_lookback_num = input_lookback_num
        self._use_soft_label = use_soft_label
        self._position_threshold = position_threshold

        self._df = self._load_data(data_fp, date_range, resample_rule, extrema_columns)
        self._df = self._create_price_columns(
            self._df,
            price_enter_column,
            price_exit_column,
            predict_lookahead_num,
            dropna_price_exit=True,
            dropna_volatility=True,
        )

        extrema_data = self._df[self._df[extrema_columns].any(axis=1)]
        extrema_num_idx = self._df.index.get_indexer(extrema_data.index)
        self._extrema_indices = [num_idx for num_idx in extrema_num_idx if num_idx >= self._input_lookback_num - 1]

    @property
    def _columns_to_use(self) -> list[str]:
        return list(
            set([*self._input_columns, self._price_enter_column, self._price_exit_column, *self._extrema_columns])
        )

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def _filter_out_nan(self, data: pd.DataFrame, columns_to_use: list[str]) -> pd.DataFrame:
        return data.dropna(subset=set(columns_to_use)).reset_index().set_index("time")

    def _load_data(
        self, data_fp: str | Path, date_range: tuple[str, str], resample_rule: str, extrema_columns: list[str]
    ) -> pd.DataFrame:
        data = pd.read_parquet(data_fp)
        data = self._filter_out_nan(data, self._columns_to_use)
        data = data[data.resample_rule == resample_rule].copy()
        time = data.index

        start, end = date_range
        logging.info(f"Using data from ({start}) to ({end})")

        data = data[(start <= time) & (time <= end)]
        return data

    def _create_price_columns(
        self,
        df: pd.DataFrame,
        price_enter_column: str,
        price_exit_column: str,
        predict_lookahead_num: int,
        dropna_price_exit: bool,
        dropna_volatility: bool,
    ) -> pd.DataFrame:
        drop_subset = []

        df[DataKey.PRICE_EXIT] = (
            df[price_exit_column].groupby(df.index.date).transform(lambda x: x.shift(-predict_lookahead_num))
        )

        df[DataKey.PRICE_ENTER] = (
            df[price_enter_column].groupby(df.index.date).transform(lambda x: x.shift(-1, fill_value=x.iloc[-1]))
        )

        df[DataKey.PRICE_ENTER_VOLATILITY_50] = df[DataKey.PRICE_ENTER].diff().abs().rolling(50).mean()
        df[DataKey.PRICE_MOVE] = df[DataKey.PRICE_EXIT] - df[DataKey.PRICE_ENTER]

        if dropna_price_exit:
            drop_subset.append(DataKey.PRICE_EXIT)
        if dropna_volatility:
            drop_subset.append(DataKey.PRICE_ENTER_VOLATILITY_50)

        if len(drop_subset) > 0:
            df = df.dropna(subset=drop_subset).copy()

        return df

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        extrema_idx = self._extrema_indices[idx]
        start_idx = extrema_idx - self._input_lookback_num + 1
        end_idx = extrema_idx + 1

        rows = self._df.iloc[start_idx:end_idx]
        current_row = rows.iloc[-1]

        candle_inputs = torch.tensor(rows[self._input_columns].values.astype(np.float32), dtype=torch.float32)

        label_val = current_row[DataKey.PRICE_MOVE]
        if self._use_soft_label:
            label_val = self._create_soft_label(label_val, self._position_threshold, tau=1.0)

        price_exit_label = torch.tensor(label_val, dtype=torch.float32)
        timestamp = torch.tensor(current_row.name.value, dtype=torch.int64)
        return candle_inputs, price_exit_label, timestamp

    def __len__(self) -> int:
        return len(self._extrema_indices)

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
