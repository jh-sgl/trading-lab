from abc import ABC, abstractmethod
from typing import Any

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

        result_df[self._target_col_name, "delta"] = delta

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
                result_df[self._target_col_name, subcol] = func()

        return result_df
