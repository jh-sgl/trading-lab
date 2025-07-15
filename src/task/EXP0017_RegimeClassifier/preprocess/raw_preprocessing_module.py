from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class RawPreprocessingModuleBase(ABC):
    @abstractmethod
    def __call__(self, result_df: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        pass


class TimeConverter(RawPreprocessingModuleBase):
    def __init__(self, time_col_name: str, date_col_name: str, set_index_and_sort: bool = True) -> None:
        self._time_col_name = time_col_name
        self._date_col_name = date_col_name
        self._set_index_and_sort = set_index_and_sort

    def __call__(self, result_df: pd.DataFrame) -> pd.DataFrame:
        date_str = result_df.fp.name.split(".")[0]
        date = pd.to_datetime(date_str, format="%Y-%m-%d")

        result_df[self._time_col_name] = pd.to_datetime(result_df[self._time_col_name].astype(str), format="%H%M%S.%f")
        result_df[self._time_col_name] = pd.to_timedelta(result_df[self._time_col_name].dt.strftime("%H:%M:%S.%f"))
        result_df[self._time_col_name] = date + result_df[self._time_col_name]
        result_df[self._date_col_name] = result_df[self._time_col_name].dt.date

        if self._set_index_and_sort:
            result_df = result_df.set_index(self._time_col_name, drop=False)
            result_df = result_df.sort_index()
        return result_df.copy()


class VolumeExtractor(RawPreprocessingModuleBase):
    def __init__(
        self,
        source_long_col_name: str | tuple[str, ...],
        source_short_col_name: str | tuple[str, ...],
        target_col_name: str,
        target_cumulative_col_name: str,
    ) -> None:
        self._source_long_col_name = source_long_col_name
        self._source_short_col_name = source_short_col_name
        self._target_col_name = target_col_name
        self._target_cumulative_col_name = target_cumulative_col_name

    def __call__(self, result_df: pd.DataFrame) -> pd.DataFrame:
        cumulative_volume = result_df[self._source_long_col_name] + result_df[self._source_short_col_name]
        result_df[self._target_cumulative_col_name] = cumulative_volume
        result_df[self._target_col_name] = cumulative_volume.diff().fillna(cumulative_volume.iloc[0])
        return result_df.copy()


class CumulativeTradeExtractor(RawPreprocessingModuleBase):
    def __init__(self, source_long_col_name: str, source_short_col_name: str, target_col_name: str) -> None:
        self._source_long_col_name = source_long_col_name
        self._source_short_col_name = source_short_col_name
        self._target_col_name = target_col_name

    def __call__(self, result_df: pd.DataFrame) -> pd.DataFrame:
        result_df[self._target_col_name] = (
            result_df[self._source_long_col_name] - result_df[self._source_short_col_name]
        )
        return result_df.copy()


class DiffExtractor(RawPreprocessingModuleBase):
    def __init__(self, source_col_name: str, target_col_name: str) -> None:
        self._source_col_name = source_col_name
        self._target_col_name = target_col_name

    def __call__(self, result_df: pd.DataFrame) -> pd.DataFrame:
        result_df[self._target_col_name] = (
            result_df[self._source_col_name].diff().fillna(result_df[self._source_col_name].iloc[0])
        )
        return result_df.copy()
