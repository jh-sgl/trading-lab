import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class BasicDataset(Dataset):
    def __init__(
        self,
        data_fp: str | Path,
        input_columns: list[str],
        label_columns: list[str],
        hold_threshold: float,
        softmax_tau: float,
        date_range: tuple[str, str],  # in "HHHH-MM-DD" format
        resample_rule: str,
        lookback_window: int,
    ) -> None:
        self._data = self._load_data(data_fp, date_range, resample_rule)
        self._input_columns = input_columns
        self._label_columns = label_columns
        self._hold_threshold = hold_threshold
        self._softmax_tau = softmax_tau
        self._lookback_window = lookback_window

    def _load_data(self, data_fp: str | Path, date_range: tuple[str, str], resample_rule: str) -> pd.DataFrame:
        data = pd.read_parquet(data_fp)
        data = data[data.resample_rule == resample_rule]
        time = pd.to_datetime(data.time)

        start, end = date_range
        logging.info(f"Using data from ({start}) to ({end})")

        data = data[(start <= time) & (time <= end)].sort_values("time")
        data = data.reset_index(drop=True)
        return data

    def _prepare_input(self, rows: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        input_ = torch.tensor(rows[self._input_columns].values, dtype=torch.float)
        input_timestamp = torch.tensor(rows["time"].apply(lambda ts: ts.timestamp()).values)
        return input_, input_timestamp

    def _prepare_label(
        self, row: pd.Series
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:

        def _create_softmax_label(label_val: float) -> torch.Tensor:
            pos_label_val = abs(label_val)
            if label_val > self._hold_threshold:
                label_ = [0, self._hold_threshold, pos_label_val]
            elif label_val < -self._hold_threshold:
                label_ = [pos_label_val, self._hold_threshold, 0]
            else:
                label_ = [0, self._hold_threshold, 0]
            label = np.array(label_).astype(np.float32)

            exp_x = np.exp((label - np.max(label)) / self._softmax_tau)
            label = exp_x / exp_x.sum(axis=-1, keepdims=True)
            return label

        label = _create_softmax_label(row[self._label_columns].item())
        label_timestamp = torch.tensor(row["time"].timestamp())
        label_price_open = torch.tensor(row["future_price_open"])
        label_price_close = torch.tensor(row["future_price_close"])
        return label, label_timestamp, label_price_open, label_price_close

    def __getitem__(self, idx: int) -> tuple[
        torch.Tensor,
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        input_, input_timestamp = self._prepare_input(self._data.iloc[idx : idx + self._lookback_window])

        label, label_timestamp, label_price_open, label_price_close = self._prepare_label(
            self._data.iloc[idx + self._lookback_window + 1]
        )

        return (
            input_,
            label,
            input_timestamp,
            label_timestamp,
            label_price_open,
            label_price_close,
        )

    def __len__(self) -> int:
        return len(self._data) - self._lookback_window - 1  # subtract 1 for label indexing


class ConsecutiveCandleDataset(BasicDataset):
    def __init__(
        self,
        data_fp: str | Path,
        input_columns: list[str],
        label_columns: list[str],
        hold_threshold: float,
        softmax_tau: float,
        date_range: tuple[str, str],  # in "HHHH-MM-DD" format
        resample_rule: str,
        lookback_window: int,
        consecutive_n: int,
    ) -> None:
        super().__init__(
            data_fp,
            input_columns,
            label_columns,
            hold_threshold,
            softmax_tau,
            date_range,
            resample_rule,
            lookback_window,
        )
        self._consecutive_n = consecutive_n
        self._data = self._add_consecutive_candles(self._data)
        self._idx_lookup = self._create_idx_lookup_for_consecutive_candles(self._data, consecutive_n)

    def _add_consecutive_candles(self, data: pd.DataFrame) -> pd.DataFrame:
        def _calc_bool_streak(flag: pd.Series) -> pd.Series:
            start_flag = flag & ~flag.shift(fill_value=False)
            group = start_flag.cumsum()
            group[~flag] = np.nan

            streak = flag.groupby(group).cumcount() + 1
            streak = streak.where(flag, 0)
            return streak

        bear_streak = _calc_bool_streak(data.future_price_delta < 0)
        bull_streak = _calc_bool_streak(data.future_price_delta > 0)
        data["delta_streak"] = np.select([bear_streak > 0, bull_streak > 0], [-1 * bear_streak, bull_streak], default=0)

        return data

    def _create_idx_lookup_for_consecutive_candles(self, data: pd.DataFrame, consecutive_n: int) -> pd.Index:
        is_n_consecutive = abs(data["delta_streak"]) >= consecutive_n
        n_consecutive_index = data[is_n_consecutive].index
        mask = (n_consecutive_index - self._lookback_window + 1 >= 0) & (n_consecutive_index + 1 < len(data))
        idx_lookup = n_consecutive_index[mask]
        return idx_lookup

    def __len__(self) -> int:
        return len(self._idx_lookup)

    def __getitem__(self, idx: int) -> tuple[
        torch.Tensor,
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        idx_for_consecutive_candles = self._idx_lookup[idx]

        input_rows = self._data.iloc[
            idx_for_consecutive_candles - self._lookback_window + 1 : idx_for_consecutive_candles + 1
        ]
        label_row = self._data.iloc[idx_for_consecutive_candles + 1]

        input_, input_timestamp = self._prepare_input(input_rows)
        label, label_timestamp, label_price_open, label_price_close = self._prepare_label(label_row)

        return (
            input_,
            label,
            input_timestamp,
            label_timestamp,
            label_price_open,
            label_price_close,
        )
