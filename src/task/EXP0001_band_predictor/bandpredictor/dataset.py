import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from util.registry import register_dataset


@dataclass
class PastInputs:
    candle_inputs: torch.Tensor
    past_band_center: torch.Tensor
    past_band_upperbound: torch.Tensor
    past_band_lowerbound: torch.Tensor


@dataclass
class TodayLabels:
    today_band_center: torch.Tensor
    today_band_upperbound: torch.Tensor
    today_band_lowerbound: torch.Tensor


@dataclass
class MetaInfo:
    today_band_offset: float
    today_timestamp: int  # ordinal
    today_cutoff_mean_price: float


@dataclass
class DataItem:
    past_inputs: PastInputs
    today_labels: TodayLabels
    meta_info: MetaInfo


@register_dataset("BandPredictor")
class BandPredictorDataset(Dataset):
    def __init__(
        self,
        data_fp: str | Path,
        input_columns: list[str],
        main_price_columns: list[str],
        band_top_column: str,
        band_bottom_column: str,
        band_offset_column: str,
        date_range: tuple[str, str],  # in "HHHH-MM-DD" format
        resample_rule: str,
        lookback_days: int,
        today_cutoff_hour: int,
        max_seq_len: int,
    ) -> None:

        self._input_columns = input_columns
        self._main_price_columns = list(main_price_columns)

        self._band_top_column = band_top_column
        self._band_bottom_column = band_bottom_column
        self._band_offset_column = band_offset_column

        self._lookback_days = lookback_days
        self._today_cutoff_hour = today_cutoff_hour
        self._max_seq_len = max_seq_len

        self._columns_to_use = (
            self._input_columns
            + self._main_price_columns
            + [self._band_top_column, self._band_bottom_column, self._band_offset_column]
        )

        self._df, self._dates = self._load_data(data_fp, date_range, resample_rule)

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def _filter_out_nan(self, data: pd.DataFrame, columns_to_use: list[str]) -> pd.DataFrame:
        return data.dropna(subset=set(columns_to_use)).reset_index().set_index("time")

    def _load_data(
        self, data_fp: str | Path, date_range: tuple[str, str], resample_rule: str
    ) -> tuple[pd.DataFrame, pd.Series]:
        data = pd.read_parquet(data_fp)
        data = self._filter_out_nan(data, self._columns_to_use)
        data = data[data.resample_rule == resample_rule]
        time = data.index

        start, end = date_range
        logging.info(f"Using data from ({start}) to ({end})")

        data = data[(start <= time) & (time <= end)]
        dates = data.date.unique()
        return data, dates

    def _calc_band(
        self, rows: pd.Series | pd.DataFrame, today_cutoff_mean_price: float, to_tensor: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[pd.Series, pd.Series, pd.Series]:
        group = rows.groupby("date")
        band_top, band_bottom, band_offset = (
            group[self._band_top_column].first(),
            group[self._band_bottom_column].first(),
            group[self._band_offset_column].first(),
        )
        band_upperbound = band_top + band_offset - today_cutoff_mean_price
        band_lowerbound = band_bottom + band_offset - today_cutoff_mean_price
        band_center = (band_upperbound + band_lowerbound) / 2

        if to_tensor:
            band_center, band_upperbound, band_lowerbound = [
                torch.tensor(item.values, dtype=torch.float32).squeeze()
                for item in [band_center, band_upperbound, band_lowerbound]
            ]

        return band_center, band_upperbound, band_lowerbound

    def _get_candle_inputs(
        self, today: datetime.date, lookback_start_date: pd.Series, today_cutoff_hour: int, to_tensor: bool
    ) -> torch.Tensor | pd.DataFrame:
        mask = ((self._df.date == today) & (self._df.index.hour < today_cutoff_hour)) | (
            (lookback_start_date < self._df.date) & (self._df.date < today)
        )
        candle_inputs = (
            self._df.loc[mask, self._input_columns]
            .groupby(self._df.date)
            .apply(lambda x: x - x.iloc[0])[-self._max_seq_len :]
        ).reset_index(level=1)

        # candle_inputs = self._df.loc[mask, self._input_columns][-self._max_seq_len :]
        if to_tensor:
            candle_inputs = torch.tensor(candle_inputs.values, dtype=torch.float)

        return candle_inputs

    def _prepare_input(
        self, rows_lookback: pd.DataFrame, rows_before_today_cutoff_hour: pd.DataFrame, today_cutoff_mean_price: float
    ) -> PastInputs:

        past_band_center, past_band_upperbound, past_band_lowerbound = self._calc_band(
            rows_lookback, today_cutoff_mean_price, to_tensor=True
        )
        rows_before_today_cutoff_hour[self._main_price_columns] -= today_cutoff_mean_price
        # candle_inputs, today_cutoff_mean_price = self._get_candle_inputs(
        #     today, lookback_start_date, self._today_cutoff_hour, to_tensor=True
        # )
        candle_inputs = torch.tensor(
            rows_before_today_cutoff_hour[self._input_columns][-self._max_seq_len :].values, dtype=torch.float
        )
        return PastInputs(candle_inputs, past_band_center, past_band_upperbound, past_band_lowerbound)

    def _prepare_label(self, rows_today: datetime.date, today_cutoff_mean_price: float) -> TodayLabels:
        today_band_center, today_band_upperbound, today_band_lowerbound = self._calc_band(
            rows_today, today_cutoff_mean_price, to_tensor=True
        )
        return TodayLabels(today_band_center, today_band_upperbound, today_band_lowerbound)

    def _prepare_metainfo(self, rows_today: pd.DataFrame, rows_today_within_cutoff: pd.DataFrame) -> MetaInfo:
        today_cutoff_mean_price = rows_today_within_cutoff[self._main_price_columns].mean(axis=1).mean()
        today_band_offset = rows_today[self._band_offset_column].iloc[0].item()
        today_date = rows_today.date.iloc[0].toordinal()
        return MetaInfo(today_band_offset, today_date, today_cutoff_mean_price)

    def __getitem__(self, idx: int) -> dict[str, dict[str, torch.Tensor]]:
        today_idx = idx + self._lookback_days
        today_date = self._dates[today_idx]
        lookback_start_date = self._dates[idx]

        within_lookback_range = self._df.date >= lookback_start_date
        before_today = self._df.date < today_date
        on_today = self._df.date == today_date
        before_cutoff_hour_mask = self._df.index.hour < self._today_cutoff_hour

        rows_before_today_cutoff_hour = self._df.loc[
            (within_lookback_range & before_today) | (on_today & before_cutoff_hour_mask)
        ].copy()
        rows_lookback = rows_before_today_cutoff_hour.loc[within_lookback_range & before_today]
        rows_today_within_cutoff = rows_before_today_cutoff_hour.loc[on_today & before_cutoff_hour_mask]
        rows_today = self._df[on_today]

        meta_info = self._prepare_metainfo(rows_today, rows_today_within_cutoff)
        past_inputs = self._prepare_input(
            rows_lookback, rows_before_today_cutoff_hour, meta_info.today_cutoff_mean_price
        )
        today_labels = self._prepare_label(rows_today, meta_info.today_cutoff_mean_price)

        return asdict(DataItem(past_inputs=past_inputs, today_labels=today_labels, meta_info=meta_info))

    def __len__(self) -> int:
        return len(self._dates) - self._lookback_days
