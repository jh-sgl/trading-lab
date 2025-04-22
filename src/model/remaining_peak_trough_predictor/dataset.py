import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from util.registry import register_dataset

from .const import DataKey


@register_dataset("RemainingPeakTroughPredictor")
class RemainingPeakTroughPredictorDataset(Dataset):
    def __init__(
        self,
        data_fp: str,
        repr_fp: str,
        resample_rule: str,
        date_range: tuple[str, str],  # in "HHHH-MM-DD" format
        input_columns: list[str],
        price_enter_column: str,
        price_exit_column: str,
        trade_stop_hour: int,
        input_lookback_num: int,
        use_repr: bool = True,
        use_normalized_label: bool = True,
        use_soft_label: bool = False,
        soft_label_tau: float | None = None,
        position_threshold: float | None = None,
        label_clip_val: float | None = None,
    ) -> None:

        self._input_columns = input_columns
        self._price_enter_column = price_enter_column
        self._price_exit_column = price_exit_column
        self._input_lookback_num = input_lookback_num

        self._use_normalized_label = use_normalized_label
        if use_soft_label and soft_label_tau is None:
            raise ValueError("When use_soft_label=True soft_label_tau can't be None.")
        self._use_soft_label = use_soft_label
        self._soft_label_tau = soft_label_tau

        self._position_threshold = position_threshold
        self._label_clip_val = label_clip_val

        self._use_repr = use_repr
        self._df, self._repr, repr_idx = self._load_data(data_fp, date_range, resample_rule, repr_fp)
        (
            self._df,
            self._trade_idx,
            self._repr_idx,
            self._price_move_mean,
            self._price_move_std,
        ) = self._create_columns_for_trade(
            self._df, price_enter_column, price_exit_column, input_lookback_num, trade_stop_hour, repr_idx
        )

    @property
    def _columns_to_use(self) -> list[str]:
        return list(
            set(
                [
                    *self._input_columns,
                    self._price_enter_column,
                    self._price_exit_column,
                ]
            )
        )

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @property
    def price_move_mean(self) -> float:
        return self._price_move_mean

    @property
    def price_move_std(self) -> float:
        return self._price_move_std

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
        return data, repr_["repr"], repr_idx

    def _create_columns_for_trade(
        self,
        df: pd.DataFrame,
        price_enter_column: str,
        price_exit_column: str,
        input_lookback_num: int,
        trade_stop_hour: int,
        repr_idx: pd.Index,
    ) -> tuple[pd.DataFrame, list[int], list[int], float, float]:

        df[DataKey.PRICE_ENTER] = (
            df[price_enter_column].groupby(df.index.date).transform(lambda x: x.shift(-1, fill_value=x.iloc[-1]))
        )

        df[DataKey.PRICE_EXIT_BUY] = (
            df[price_exit_column].groupby(df.index.date).transform(lambda x: x.shift(-1).iloc[::-1].cummax().iloc[::-1])
        )
        df[DataKey.PRICE_EXIT_SELL] = (
            df[price_exit_column].groupby(df.index.date).transform(lambda x: x.shift(-1).iloc[::-1].cummin().iloc[::-1])
        )

        df[DataKey.PRICE_ENTER_VOLATILITY_50] = df[DataKey.PRICE_ENTER].diff().abs().rolling(50, min_periods=1).mean()
        df = df.dropna(subset=[DataKey.PRICE_ENTER_VOLATILITY_50]).copy()

        df[DataKey.PRICE_MOVE_BUY] = df[DataKey.PRICE_EXIT_BUY] - df[DataKey.PRICE_ENTER]
        df[DataKey.PRICE_MOVE_SELL] = df[DataKey.PRICE_EXIT_SELL] - df[DataKey.PRICE_ENTER]

        trade_rows = df[df[DataKey.PRICE_MOVE_BUY].notna() & df[DataKey.PRICE_MOVE_SELL].notna()]
        trade_rows = trade_rows[trade_rows.index.hour < trade_stop_hour]

        price_move_mean = trade_rows[[DataKey.PRICE_MOVE_BUY, DataKey.PRICE_MOVE_SELL]].values.mean()
        price_move_std = trade_rows[[DataKey.PRICE_MOVE_BUY, DataKey.PRICE_MOVE_SELL]].values.std()

        df[DataKey.PRICE_MOVE_BUY_NORMALIZED] = (df[DataKey.PRICE_MOVE_BUY] - price_move_mean) / (
            price_move_std + 1e-15
        )
        df[DataKey.PRICE_MOVE_SELL_NORMALIZED] = (df[DataKey.PRICE_MOVE_SELL] - price_move_mean) / (
            price_move_std + 1e-15
        )

        trade_idx = df.index.get_indexer(trade_rows.index)
        trade_idx = [t_idx for t_idx in trade_idx if (t_idx >= input_lookback_num - 1)]

        repr_idx = repr_idx.get_indexer(df.index)

        return (
            df,
            trade_idx,
            repr_idx,
            price_move_mean,
            price_move_std,
        )

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        orig_idx = self._trade_idx[idx]
        start_idx = orig_idx - self._input_lookback_num + 1
        end_idx = orig_idx + 1

        rows = self._df.iloc[start_idx:end_idx]
        current_row = rows.iloc[-1]

        inputs_to_concat = []
        if len(self._input_columns) > 0:
            inputs_to_concat.append(rows[self._input_columns].values)

        if self._use_repr:
            repr_start_idx = self._repr_idx[start_idx]
            repr_end_idx = self._repr_idx[end_idx]
            repr_inputs = self._repr[repr_start_idx:repr_end_idx]
            repr_inputs = repr_inputs.reshape(repr_inputs.shape[0], -1)
            inputs_to_concat.append(repr_inputs)

        inputs = torch.tensor(np.concat(inputs_to_concat, axis=-1), dtype=torch.float32)

        if self._use_normalized_label:
            label_buy = current_row[DataKey.PRICE_MOVE_BUY_NORMALIZED]
            label_sell = current_row[DataKey.PRICE_MOVE_SELL_NORMALIZED]
        else:
            label_buy = current_row[DataKey.PRICE_MOVE_BUY]
            label_sell = current_row[DataKey.PRICE_MOVE_SELL]

        if self._label_clip_val is not None:
            if abs(label_buy) < self._label_clip_val:
                label_buy = np.float64(0)
            if abs(label_sell) < self._label_clip_val:
                label_sell = np.float64(0)

        if self._use_soft_label:
            label_buy = self._create_soft_label(label_buy, self._position_threshold, tau=self._soft_label_tau)
            label_sell = self._create_soft_label(label_sell, self._position_threshold, tau=self._soft_label_tau)

        label = np.stack([label_buy, label_sell])
        price_exit_label = torch.tensor(label, dtype=torch.float32)
        timestamp = torch.tensor(current_row.name.value, dtype=torch.int64)
        return inputs, price_exit_label, timestamp

    def __len__(self) -> int:
        return len(self._trade_idx)

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
