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

    @property
    @abstractmethod
    def target_col_names(self) -> list[tuple[str, str]]:
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

    @property
    def target_col_names(self) -> list[tuple[str, str]]:
        return [(self._date_col_name, "__NA__")]


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
        computed = {}
        for key in self._select_ohlc:
            if key not in ohlcv_ops:
                raise ValueError(f"Unsupported OHLCV Key: '{key}'")

            subcol, func = ohlcv_ops[key]
            col_name = (self._target_col_name, subcol)
            computed[col_name] = func()

        computed_df = pd.DataFrame(computed, index=result_df.index)
        result_df = pd.concat([result_df, computed_df], axis=1)

        return result_df

    @property
    def target_col_names(self) -> list[tuple[str, str]]:
        target_cols = []
        for key in self._select_ohlc:
            target_cols.append((self._target_col_name, key))
        return target_cols
