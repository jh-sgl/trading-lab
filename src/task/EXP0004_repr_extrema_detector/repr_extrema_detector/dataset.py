import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from util.registry import register_dataset

from .const import DataKey


@register_dataset("ReprExtremaDetector")
class ReprExtremaDetectorDataset(Dataset):
    def __init__(
        self,
        data_fp: str | Path,
        repr_fp: str,
        resample_rule: str,
        sharpe_column: str,
        price_enter_column: str,
        price_exit_column: str,
        date_range: tuple[str, str],  # in "HHHH-MM-DD" format
        input_lookback_num: int,
        predict_lookahead_num: int,
        use_soft_label: bool = False,
        position_threshold: float | None = None,
    ) -> None:

        self._price_enter_column = price_enter_column
        self._price_exit_column = price_exit_column
        self._input_lookback_num = input_lookback_num
        self._predict_lookahead_num = predict_lookahead_num
        self._use_soft_label = use_soft_label
        self._position_threshold = position_threshold

        self._df, self._repr, self._repr_idx = self._load_data(data_fp, date_range, resample_rule, repr_fp)
        self._df, valid_index = self._create_columns_for_trading(
            self._df,
            sharpe_column,
            price_enter_column,
            price_exit_column,
            predict_lookahead_num,
        )
        valid_num_idx = self._df.index.get_indexer(valid_index)
        self._valid_index = [num_idx for num_idx in valid_num_idx if num_idx >= self._input_lookback_num - 1]

    @property
    def _columns_to_use(self) -> list[str]:
        return list(set([self._price_enter_column, self._price_exit_column]))

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def _filter_out_nan(self, data: pd.DataFrame, columns_to_use: list[str]) -> pd.DataFrame:
        return data.dropna(subset=set(columns_to_use)).reset_index().set_index("time")

    def _load_data(
        self, data_fp: str | Path, date_range: tuple[str, str], resample_rule: str, repr_fp: str
    ) -> tuple[pd.DataFrame, torch.Tensor, Callable]:
        orig_data = pd.read_parquet(data_fp)

        data = self._filter_out_nan(orig_data, self._columns_to_use)
        data = data[data.resample_rule == resample_rule].copy()
        time = data.index

        start, end = date_range
        logging.info(f"Using data from ({start}) to ({end})")

        data = data[(start <= time) & (time <= end)]

        repr_ = torch.load(repr_fp, weights_only=False)
        repr_idx = pd.to_datetime(repr_["df"]["time"]["open"].dt.floor("5T").values)

        data = data.loc[repr_idx[repr_idx.isin(data.index)]].copy()
        repr_idx = repr_idx.get_indexer(data.index)
        return data, repr_["repr"], repr_idx

    def _create_columns_for_trading(
        self,
        df: pd.DataFrame,
        sharpe_column: str,
        price_enter_column: str,
        price_exit_column: str,
        predict_lookahead_num: int,
    ) -> tuple[pd.DataFrame, pd.Index]:

        df[DataKey.PRICE_EXIT] = (
            df[price_exit_column].groupby(df.index.date).transform(lambda x: x.shift(-predict_lookahead_num))
        )

        df[DataKey.PRICE_ENTER] = (
            df[price_enter_column].groupby(df.index.date).transform(lambda x: x.shift(-1, fill_value=x.iloc[-1]))
        )

        df[DataKey.PRICE_ENTER_VOLATILITY_50] = df[DataKey.PRICE_ENTER].diff().abs().rolling(50, min_periods=1).mean()
        df[DataKey.PRICE_MOVE] = df[DataKey.PRICE_EXIT] - df[DataKey.PRICE_ENTER]
        # df[DataKey.BIDIRECTIONAL_SHARPE] = df[sharpe_column].shift(-predict_lookahead_num + 1) * (252**0.5)
        df[DataKey.BIDIRECTIONAL_SHARPE] = (df[sharpe_column].shift(-predict_lookahead_num + 1) - df[sharpe_column]) * (
            252**0.5
        )
        df[DataKey.BIDIRECTIONAL_SHARPE] = (
            df[DataKey.BIDIRECTIONAL_SHARPE] - df[DataKey.BIDIRECTIONAL_SHARPE].mean()
        ) / df[DataKey.BIDIRECTIONAL_SHARPE].std()

        df = df.dropna(subset=[DataKey.BIDIRECTIONAL_SHARPE])
        valid_index = df[df[DataKey.PRICE_MOVE].notna()].index
        return df, valid_index

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start_idx = self._valid_index[idx] - self._input_lookback_num + 1
        end_idx = self._valid_index[idx] + 1

        rows = self._df.iloc[start_idx:end_idx]
        current_row = rows.iloc[-1]

        repr_start_idx = self._repr_idx[start_idx]
        repr_end_idx = self._repr_idx[end_idx]
        repr_inputs = self._repr[repr_start_idx:repr_end_idx]
        repr_inputs = repr_inputs.reshape(*repr_inputs.shape[:-2], -1)
        label_val = current_row[DataKey.BIDIRECTIONAL_SHARPE]

        if self._use_soft_label:
            label_val = self._create_soft_label(label_val, self._position_threshold, tau=2.5)

        price_exit_label = torch.tensor(label_val, dtype=torch.float32)
        timestamp = torch.tensor(current_row.name.value, dtype=torch.int64)
        return repr_inputs, price_exit_label, timestamp

    def __len__(self) -> int:
        return len(self._valid_index)

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
