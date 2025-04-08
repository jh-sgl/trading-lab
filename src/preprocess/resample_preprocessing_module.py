from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from pandas.core.resample import Resampler


class ResamplePreprocessingModuleBase(ABC):
    @abstractmethod
    def __call__(
        self,
        result_df: pd.DataFrame,
        source_df_resampled: Resampler,
        *args: Any,
        **kwargs: Any,
    ) -> pd.DataFrame:
        pass


class DateExtractor(ResamplePreprocessingModuleBase):
    def __init__(self, date_col_name: str) -> None:
        self._date_col_name = date_col_name

    def __call__(
        self,
        result_df: pd.DataFrame,
        source_df_resampled: Resampler,
    ) -> pd.DataFrame:
        result_df[self._date_col_name] = result_df.index.date
        return result_df


class DeltaExtractor(ResamplePreprocessingModuleBase):
    def __init__(
        self,
        source_col_name: str | tuple[str, ...],
        target_col_name: str,
    ) -> None:
        self._source_col_name = source_col_name
        self._target_col_name = target_col_name

    def __call__(
        self,
        result_df: pd.DataFrame,
        source_df_resampled: Resampler,
    ) -> pd.DataFrame:
        source_series = source_df_resampled[self._source_col_name]

        open_price = source_series.first()
        close_price = source_series.last()
        delta = close_price - open_price

        result_df[self._target_col_name + "_delta"] = delta

        return result_df


class OHLCVExtractor(ResamplePreprocessingModuleBase):
    def __init__(
        self,
        source_col_name: str | tuple[str, ...],
        target_col_name: str,
        select_ohlc: str = "ohlc",
    ) -> None:
        self._source_col_name = source_col_name
        self._target_col_name = target_col_name
        self._select_ohlc = select_ohlc

    def __call__(
        self,
        result_df: pd.DataFrame,
        source_df_resampled: Resampler,
    ) -> pd.DataFrame:

        source_series = source_df_resampled[self._source_col_name]

        ohlcv_ops = {
            "o": ("open", source_series.first),
            "h": ("high", source_series.max),
            "l": ("low", source_series.min),
            "c": ("close", source_series.last),
            "v": (None, source_series.sum),
        }

        for key in self._select_ohlc:
            if key not in ohlcv_ops:
                raise ValueError(f"Unsupported OHLCV Key: '{key}'")

            subcol, func = ohlcv_ops[key]
            if subcol is None:
                result_df[self._target_col_name] = func()
            else:
                result_df[self._target_col_name + "_" + subcol] = func()

        return result_df


class MaxCrossingBandExtractor(ResamplePreprocessingModuleBase):
    def __init__(
        self,
        source_col_name: str,
        target_col_name: str,
        date_col_name: str,
        band_width: float,
    ) -> None:
        self._open_col_name = source_col_name + "_open"
        self._high_col_name = source_col_name + "_high"
        self._low_col_name = source_col_name + "_low"
        self._close_col_name = source_col_name + "_close"
        self._ohlc_col_names = [
            self._open_col_name,
            self._high_col_name,
            self._low_col_name,
            self._close_col_name,
        ]

        self._date_col_name = date_col_name

        self._max_crossing_band_count_col_name = target_col_name + "_max_crossing_band_count"
        self._max_crossing_band_top_col_name = target_col_name + "_max_crossing_band_top"
        self._max_crossing_band_bottom_col_name = target_col_name + "_max_crossing_band_bottom"
        self._max_crossing_band_offset_tr_col_name = target_col_name + "_max_crossing_band_offset_tr"

        self._band_width = band_width

    def _check_col_names(self, result_df: pd.DataFrame) -> None:
        not_found_col_names = [col_name for col_name in self._ohlc_col_names if col_name not in result_df.keys()]
        if len(not_found_col_names) > 0:
            raise ValueError(f"Given column names are not found in result_df: {not_found_col_names}")

    def _calculate_band_crossing_max(self, daily_df: pd.DataFrame) -> pd.Series:
        signal_max = daily_df[self._ohlc_col_names].max(axis=1)
        signal_min = daily_df[self._ohlc_col_names].min(axis=1)

        bands = np.arange(signal_min.min(), signal_max.max(), self._band_width)

        max_crossing_count = 0
        max_crossing_band_bottom, max_crossing_band_top = None, None
        for b_bottom, b_top in zip(bands, bands[1:]):
            cnt = 0
            for smin, smax in zip(signal_min, signal_max):
                if (smin < b_bottom) and (b_top < smax):
                    cnt += 1
            if cnt > max_crossing_count:
                max_crossing_count = cnt
                max_crossing_band_bottom, max_crossing_band_top = b_bottom, b_top

        high = daily_df[self._high_col_name]
        low = daily_df[self._low_col_name]
        close = daily_df[self._close_col_name]

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)
        tr = tr.mean()

        return pd.Series(
            {
                self._max_crossing_band_count_col_name: max_crossing_count,
                self._max_crossing_band_bottom_col_name: max_crossing_band_bottom,
                self._max_crossing_band_top_col_name: max_crossing_band_top,
                self._max_crossing_band_offset_tr_col_name: tr,
            }
        )

    def __call__(
        self,
        result_df: pd.DataFrame,
        source_df_resampled: Resampler,
    ) -> pd.DataFrame:
        self._check_col_names(result_df)

        df = result_df[self._ohlc_col_names]

        band_crossing_df = df.groupby(df.index.date).apply(lambda x: self._calculate_band_crossing_max(x))

        result_df = pd.merge(result_df, band_crossing_df, how="left", left_on=self._date_col_name, right_index=True)
        return result_df


class MarketClosingPriceAssigner(ResamplePreprocessingModuleBase):
    def __init__(self, source_col_name: str, target_col_name: str) -> None:
        self._source_col_name = source_col_name
        self._target_col_name = target_col_name

    def _check_col_name(self, result_df: pd.DataFrame) -> None:
        if self._source_col_name not in result_df.keys():
            raise ValueError(f"Given source_col_name is not found in result_df: {self._source_col_name}")

    def __call__(self, result_df: pd.DataFrame, source_df_resampled: Resampler) -> pd.DataFrame:
        self._check_col_name(result_df)
        result_df[self._target_col_name] = result_df[self._source_col_name].iloc[-1]
        return result_df
