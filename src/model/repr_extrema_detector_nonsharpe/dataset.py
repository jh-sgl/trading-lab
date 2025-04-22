import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from util.registry import register_dataset

from .const import DataKey


@register_dataset("ReprExtremaDetectorNonsharpe")
class ReprExtremaDetectorNonsharpeDataset(Dataset):
    def __init__(
        self,
        data_fp: str | Path,
        repr_fp: str,
        resample_rule: str,
        minima_column: str,
        maxima_column: str,
        price_enter_column: str,
        price_exit_column: str,
        exit_strategy: str,
        date_range: tuple[str, str],  # in "HHHH-MM-DD" format
        input_lookback_num: int,
        predict_lookahead_num: int,
        use_soft_label: bool = False,
        position_threshold: float | None = None,
    ) -> None:

        self._minima_column = minima_column
        self._maxima_column = maxima_column
        self._price_enter_column = price_enter_column
        self._price_exit_column = price_exit_column
        self._input_lookback_num = input_lookback_num
        self._use_soft_label = use_soft_label
        self._position_threshold = position_threshold

        self._df, self._repr, self._repr_idx = self._load_data(data_fp, date_range, resample_rule, repr_fp)
        self._df = self._create_price_columns(
            self._df,
            price_enter_column,
            price_exit_column,
            minima_column,
            maxima_column,
            exit_strategy,
            predict_lookahead_num,
        )

        extrema_data = self._df[self._df[[minima_column, maxima_column]].any(axis=1)]
        extrema_num_idx = self._df.index.get_indexer(extrema_data.index)
        self._extrema_indices = [
            num_idx
            for num_idx in extrema_num_idx
            if (num_idx >= self._input_lookback_num - 1) and not np.isnan(self._df.iloc[num_idx][DataKey.PRICE_MOVE])
        ]

    @property
    def _columns_to_use(self) -> list[str]:
        return list(
            set(
                [
                    self._price_enter_column,
                    self._price_exit_column,
                    self._minima_column,
                    self._maxima_column,
                ]
            )
        )

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def _filter_out_nan(self, data: pd.DataFrame, columns_to_use: list[str]) -> pd.DataFrame:
        return data.dropna(subset=set(columns_to_use)).reset_index().set_index("time")

    def _load_data(
        self, data_fp: str | Path, date_range: tuple[str, str], resample_rule: str, repr_fp: str
    ) -> tuple[pd.DataFrame, torch.Tensor, pd.Index]:
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

    def _create_next_opposite_extrema_price(
        self, df: pd.DataFrame, minima_column: str, maxima_column: str, price_exit_column: str, periods: int
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

        last_price = df[price_exit_column].iloc[-1]
        if len(minima_next_maxima) > 0:
            for src, tgt in zip(minima_idx, minima_next_maxima):
                result_series.iloc[src] = df.iloc[tgt][price_exit_column] if tgt != -1 else last_price
        if len(maxima_next_minima) > 0:
            for src, tgt in zip(maxima_idx, maxima_next_minima):
                result_series.iloc[src] = df.iloc[tgt][price_exit_column] if tgt != -1 else last_price

        return result_series

    def _create_price_columns(
        self,
        df: pd.DataFrame,
        price_enter_column: str,
        price_exit_column: str,
        minima_column: str,
        maxima_column: str,
        exit_strategy: str,
        predict_lookahead_num: int,
    ) -> pd.DataFrame:

        if exit_strategy == "next_opposite_extrema":
            df[DataKey.PRICE_EXIT] = (
                df.groupby(df.index.date)
                .apply(
                    lambda x: self._create_next_opposite_extrema_price(
                        x, minima_column, maxima_column, price_exit_column, periods=-1
                    )
                )
                .reset_index(level=0, drop=True)
            )
        elif exit_strategy == "predict_lookahead":
            df[DataKey.PRICE_EXIT] = (
                df[price_exit_column].groupby(df.index.date).transform(lambda x: x.shift(-predict_lookahead_num))
            )

        df[DataKey.PRICE_ENTER] = (
            df[price_enter_column].groupby(df.index.date).transform(lambda x: x.shift(-1, fill_value=x.iloc[-1]))
        )

        df[DataKey.PRICE_ENTER_VOLATILITY_50] = df[DataKey.PRICE_ENTER].diff().abs().rolling(50, min_periods=1).mean()

        df[DataKey.PRICE_MOVE] = df[DataKey.PRICE_EXIT] - df[DataKey.PRICE_ENTER]
        df[DataKey.PRICE_MOVE] = (df[DataKey.PRICE_MOVE] - df[DataKey.PRICE_MOVE].mean()) / df[DataKey.PRICE_MOVE].std()

        # if exit_strategy == "next_opposite_extrema":
        #     df.loc[df[maxima_column], DataKey.PRICE_MOVE] *= -1

        # if dropna_price_exit:
        #     drop_subset.append(DataKey.PRICE_EXIT)
        # if dropna_volatility:

        df = df.dropna(subset=[DataKey.PRICE_ENTER_VOLATILITY_50]).copy()

        return df

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        extrema_idx = self._extrema_indices[idx]
        start_idx = extrema_idx - self._input_lookback_num + 1
        end_idx = extrema_idx + 1

        rows = self._df.iloc[start_idx:end_idx]
        current_row = rows.iloc[-1]

        repr_start_idx = self._repr_idx[start_idx]
        repr_end_idx = self._repr_idx[end_idx]
        repr_inputs = self._repr[repr_start_idx:repr_end_idx]
        repr_inputs = repr_inputs.reshape(-1)

        label_val = current_row[DataKey.PRICE_MOVE]
        if self._use_soft_label:
            label_val = self._create_soft_label(label_val, self._position_threshold, tau=2.5)

        price_exit_label = torch.tensor(label_val, dtype=torch.float32)
        timestamp = torch.tensor(current_row.name.value, dtype=torch.int64)
        return repr_inputs, price_exit_label, timestamp

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
