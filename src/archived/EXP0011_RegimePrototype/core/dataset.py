import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numba import njit
from torch.utils.data import Dataset

from util.const import Num
from util.registry import register_dataset

from .const import DFKey


@njit
def _compute_series_profit_trajectory(price_arr, lookahead, commission_rate, slippage_per_exec):
    slippage = slippage_per_exec * 2

    n = len(price_arr)
    long_traj = np.full((n, lookahead), np.nan)
    short_traj = np.full((n, lookahead), np.nan)
    long_traj_costless = np.full((n, lookahead), np.nan)
    short_traj_costless = np.full((n, lookahead), np.nan)
    hold_traj = np.zeros((n, lookahead))  # always zero

    for i in range(n - lookahead):
        window = price_arr[i : i + lookahead]
        open_price = price_arr[i]
        close_prices = window

        commission = (open_price + close_prices) * commission_rate

        long_profit_costless = close_prices - open_price
        short_profit_costless = open_price - close_prices
        long_profit = long_profit_costless - commission - slippage
        short_profit = short_profit_costless - commission - slippage

        long_traj_costless[i, :] = long_profit_costless
        short_traj_costless[i, :] = short_profit_costless
        long_traj[i, :] = long_profit
        short_traj[i, :] = short_profit
    return long_traj, short_traj, long_traj_costless, short_traj_costless, hold_traj


@register_dataset("basic")
class BasicDataset(Dataset):
    def __init__(
        self,
        data_fp: str,
        date_range: tuple[str, str],  # in "HHHH-MM-DD" format
        resample_rule: str,
        input_columns_info: list[str],
        lookback_num_longterm: int,
        lookback_num_shortterm: int,
        lookahead_num: int,
    ) -> None:
        self._input_columns_info = input_columns_info
        self._lookback_num_longterm = lookback_num_longterm
        self._lookback_num_shortterm = lookback_num_shortterm
        self._lookahead_num = lookahead_num

        self._df = self._load_data(data_fp, date_range, resample_rule)
        self._df, self._dataloader_idx, self._traj = self._create_columns_for_trade(
            self._df, lookback_num_longterm, lookahead_num
        )

    def _compute_profit_trajectory(self, df: pd.DataFrame, lookahead_num: int) -> pd.DataFrame:
        results = []

        for date, group in df.groupby(df.index.date):
            price_series = group[DFKey.PRICE_EXECUTION].to_numpy(dtype=np.float64)
            long_traj, short_traj, long_traj_costless, short_traj_costless, hold_traj = (
                _compute_series_profit_trajectory(
                    price_series,
                    lookahead=lookahead_num,
                    commission_rate=Num.COMMISSION_RATE,
                    slippage_per_exec=Num.SLIPPAGE_PER_EXECUTION,
                )
            )
            traj_df = pd.DataFrame(
                {
                    DFKey.LONG_PROFIT_TRAJECTORY: long_traj.tolist(),
                    DFKey.SHORT_PROFIT_TRAJECTORY: short_traj.tolist(),
                    DFKey.LONG_PROFIT_TRAJECTORY_COSTLESS: long_traj_costless.tolist(),
                    DFKey.SHORT_PROFIT_TRAJECTORY_COSTLESS: short_traj_costless.tolist(),
                    DFKey.HOLD_PROFIT_TRAJECTORY: hold_traj.tolist(),
                },
                index=group.index,
            )
            group = group.copy()
            group = group.join(traj_df)
            results.append(group)
        return pd.concat(results).sort_index()

    def _create_columns_for_trade(
        self, df: pd.DataFrame, lookback_num: int, lookahead_num: int
    ) -> tuple[pd.DataFrame, list[int], torch.Tensor]:

        # execution right after seeing 'close' of the current candle = open of the next candle
        df[DFKey.PRICE_EXECUTION] = (
            df[DFKey.FUTURE_PRICE_OPEN].groupby(df.index.date).transform(lambda x: x.shift(-1, fill_value=x.iloc[-1]))
        )
        df = self._compute_profit_trajectory(df, lookahead_num)

        df[DFKey.VOLATILITY_50] = df[DFKey.PRICE_EXECUTION].diff().abs().rolling(50, min_periods=1).mean()

        has_traj = df[DFKey.LONG_PROFIT_TRAJECTORY].apply(lambda x: False if np.isnan(x[0]) else True)
        mask = has_traj & (np.arange(len(df)) >= (lookback_num - 1))
        dataloader_idx = np.flatnonzero(mask).tolist()

        traj = []
        for _, row in df.loc[
            has_traj, [DFKey.SHORT_PROFIT_TRAJECTORY, DFKey.HOLD_PROFIT_TRAJECTORY, DFKey.LONG_PROFIT_TRAJECTORY]
        ].iterrows():
            traj.append(torch.from_numpy(np.stack(row.values, axis=-1, dtype=np.float32)))

        traj = torch.stack(traj)
        traj = torch.softmax(traj.view(traj.shape[0], -1), dim=-1).view_as(traj)
        return df, dataloader_idx, traj

    @property
    def _columns_to_use(self) -> list[str]:
        return self._input_columns_info.keys()

    def _filter_out_nan(self, data: pd.DataFrame, columns_to_use: list[str]) -> pd.DataFrame:
        return data.dropna(subset=set(columns_to_use)).reset_index().set_index("time")

    def _load_data(self, data_fp: str | Path, date_range: tuple[str, str], resample_rule: str) -> pd.DataFrame:
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

    def _prepare_inputs(
        self, current_idx: int, lookback_num_longterm: int, lookback_num_shortterm: int
    ) -> torch.Tensor:
        past_start = current_idx - lookback_num_longterm + 1
        past_rows = self._df.iloc[past_start : current_idx + 1]  # current-inclusive
        date_token = torch.tensor(pd.factorize(past_rows.index.date)[0], dtype=torch.float).unsqueeze(-1)

        df_inputs = past_rows[self._input_columns_info.keys()].copy()
        for col_name, normalize in self._input_columns_info.items():
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
        df_inputs = torch.tensor(df_inputs.values.astype(float), dtype=torch.float32)
        inputs_longterm = torch.cat([df_inputs, date_token], dim=-1)
        inputs_shortterm = inputs_longterm[-lookback_num_shortterm:]
        return inputs_longterm, inputs_shortterm

        # def _prepare_label(self, current_idx: int) -> torch.Tensor:
        #     current_row = self._df.iloc[current_idx]
        #     return torch.tensor(
        #         np.stack(
        #             current_row[
        #                 [DFKey.LONG_PROFIT_TRAJECTORY, DFKey.HOLD_PROFIT_TRAJECTORY, DFKey.SHORT_PROFIT_TRAJECTORY]
        #             ],
        #             axis=-1,
        #         ),
        #         dtype=torch.float32,
        #    -1)
        # return inputs

    def _prepare_timestamp(self, current_idx: int) -> torch.Tensor:
        current_row = self._df.iloc[current_idx]
        return torch.tensor(current_row.name.value, dtype=torch.int64)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        current_idx = self._dataloader_idx[idx]
        label = self._traj[idx]

        inputs_longterm, inputs_shortterm = self._prepare_inputs(
            current_idx, self._lookback_num_longterm, self._lookback_num_shortterm
        )
        timestamp = self._prepare_timestamp(current_idx)
        return inputs_longterm, inputs_shortterm, label, timestamp

    def __len__(self) -> int:
        return len(self._dataloader_idx)
