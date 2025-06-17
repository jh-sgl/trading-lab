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
        result_df[(self._date_col_name, "__NA__")] = result_df.index.date
        return result_df


class OHLCVExtractor(ResamplePreprocessingModuleBase):
    def __init__(
        self,
        source_col_name: str | tuple[str, ...],
        target_col_name: str,
        select_ohlc: str,
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
            "v": ("volume", source_series.sum),
        }

        for key in self._select_ohlc:
            if key not in ohlcv_ops:
                raise ValueError(f"Unsupported OHLCV Key: '{key}'")

            subcol, func = ohlcv_ops[key]
            result_df[self._target_col_name, subcol] = func()

        return result_df.copy()
