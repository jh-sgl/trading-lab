import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

from util.const import Num
from util.registry import register_dataset

from .const import DFKey
from .strategy import Strategy, build_strategy


@register_dataset("basic")
class BasicDataset(Dataset):
    def __init__(
        self,
        data_fp: str,
        date_range: tuple[str, str],  # in "HHHH-MM-DD" format
        strategy: DictConfig,
        resample_rule: str,
        df_input_columns_info: dict[str, str | None],
        lookback_num: int,
    ) -> None:
        self._date_range = date_range

        self._df_input_columns_info = df_input_columns_info
        self._lookback_num = lookback_num

        df = self._load_data(data_fp, date_range, resample_rule)
        df = self._filter_out_nan(df, self._df_input_columns_info.keys())

        self._strategy = build_strategy(strategy, df)
        strategy_df = self._strategy.backtest()
        self._strategy_input_columns_info, self._strategy_label_column = self._strategy.get_columns_info_for_model()
        strategy_df = self._filter_out_nan(strategy_df, self._strategy_input_columns_info.keys())
        self._strategy_df = self._strategy.update_df(strategy_df)

        self._dataloader_idx = self._select_dataloader_idx(self._strategy_df)

    def _select_dataloader_idx(self, df: pd.DataFrame) -> list[int]:
        rule_trading_mask = df[DFKey.ORIG_SIGNAL] != 0
        numerical_indices = np.flatnonzero(rule_trading_mask)
        dataloader_idx = [i for i in numerical_indices if i > self._lookback_num]
        return dataloader_idx

    def _filter_out_nan(self, data: pd.DataFrame, columns_to_use: list[str]) -> pd.DataFrame:
        return data.dropna(subset=set(columns_to_use)).reset_index().set_index("time")

    def _load_data(self, data_fp: str | Path, date_range: tuple[str, str], resample_rule: str) -> pd.DataFrame:
        orig_data = pd.read_parquet(data_fp)

        data = orig_data[orig_data.resample_rule == resample_rule]

        start, end = date_range
        time = data.index
        data = data[(start <= time) & (time <= end)].copy()

        logging.info(f"Includes data from ({start}) to ({end})")
        return data

    @property
    def date_range(self) -> tuple[str, str]:
        return self._date_range

    @property
    def strategy_df(self) -> pd.DataFrame:
        return self._strategy_df

    @property
    def strategy(self) -> Strategy:
        return self._strategy

    def _normalize_inputs(self, df_inputs: pd.DataFrame, input_columns_info: dict[str, str | None]) -> pd.DataFrame:
        for col_name, normalize in input_columns_info.items():
            if normalize == "first_row":
                df_inputs[col_name] -= df_inputs[col_name].iloc[0]
            elif normalize == "standardized_normal":
                df_inputs[col_name] = (df_inputs[col_name] - df_inputs[col_name].mean()) / (
                    df_inputs[col_name].std() + Num.EPS
                )
            elif normalize == None:
                pass
            else:
                raise ValueError(f"Unknown normalization mode: {normalize}")
        return df_inputs

    def _create_date_token(self, past_rows: pd.DataFrame) -> torch.Tensor:
        return torch.tensor(pd.factorize(past_rows.index.date)[0], dtype=torch.float32).unsqueeze(-1)

    def _prepare_inputs(self, current_idx: int) -> torch.Tensor:
        past_start = current_idx - self._lookback_num + 1
        past_rows = self._strategy_df.iloc[past_start : current_idx + 1]  # current-inclusive

        date_token = self._create_date_token(past_rows)

        df_inputs = past_rows[self._df_input_columns_info.keys()].copy()
        df_inputs = self._normalize_inputs(df_inputs, self._df_input_columns_info)
        df_inputs = torch.tensor(df_inputs.values.astype(float), dtype=torch.float32)

        strategy_inputs = past_rows[self._strategy_input_columns_info.keys()].copy()
        strategy_inputs = self._normalize_inputs(strategy_inputs, self._strategy_input_columns_info)
        strategy_inputs = torch.tensor(strategy_inputs.values.astype(float), dtype=torch.float32)

        inputs = torch.cat([df_inputs, strategy_inputs, date_token], dim=-1)
        return inputs

    def _prepare_label(self, current_idx: int) -> torch.Tensor:
        current_row = self._strategy_df.iloc[current_idx]
        pnl = current_row[self._strategy_label_column]
        orig_signal = current_row[DFKey.ORIG_SIGNAL]
        if pnl > 0:
            if orig_signal == 1:
                label = torch.tensor([0, 0.25, pnl], dtype=torch.float32)
            elif orig_signal == -1:
                label = torch.tensor([pnl, 0.25, 0], dtype=torch.float32)
        else:
            label = torch.tensor([0, 0.25, 0], dtype=torch.float32)
        return label.softmax(dim=-1)

    def _prepare_timestamp(self, current_idx: int) -> torch.Tensor:
        current_row = self._strategy_df.iloc[current_idx]
        return torch.tensor(current_row.name.value, dtype=torch.int64)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        current_idx = self._dataloader_idx[idx]

        inputs = self._prepare_inputs(current_idx)
        label = self._prepare_label(current_idx)
        timestamp = self._prepare_timestamp(current_idx)
        return inputs, label, timestamp

    def __len__(self) -> int:
        return len(self._dataloader_idx)
