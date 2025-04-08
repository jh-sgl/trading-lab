from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class AfterMergePreprocessingModuleBase(ABC):
    @abstractmethod
    def __call__(self, result_df: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        pass


class DailyShifter(AfterMergePreprocessingModuleBase):
    def __init__(self, source_col_name: str, target_col_name: str, date_col_name: str, pd_shift_args: Any) -> None:
        self._source_col_name = source_col_name
        self._target_col_name = target_col_name
        self._date_col_name = date_col_name
        self._pd_shift_args = pd_shift_args

    def __call__(self, result_df: pd.DataFrame) -> pd.DataFrame:
        daily = result_df.groupby(["resample_rule", self._date_col_name])[self._source_col_name].first().reset_index()
        daily[self._target_col_name] = daily.groupby("resample_rule")[self._source_col_name].shift(
            **self._pd_shift_args
        )
        result_df = (
            result_df.reset_index()
            .merge(
                daily[["resample_rule", self._date_col_name, self._target_col_name]],
                on=["resample_rule", self._date_col_name],
                how="left",
            )
            .set_index("time")
        )
        return result_df


class DisparityExtractor(AfterMergePreprocessingModuleBase):
    def __init__(self, source_col_name: str, target_col_name: str, rolling_window: int) -> None:
        self._source_col_name = source_col_name
        self._target_col_name = target_col_name
        self._rolling_window = rolling_window

    def __call__(self, result_df: pd.DataFrame) -> pd.DataFrame:
        disparity = (
            result_df.groupby("resample_rule")[self._source_col_name]
            .apply(lambda x: x - x.rolling(self._rolling_window, min_periods=1).mean())
            .rename(self._target_col_name)
        )
        result_df = (
            result_df.reset_index()
            .merge(disparity.reset_index(), on=["resample_rule", "time"], how="left")
            .set_index("time")
        )
        return result_df


class ExtremaExtractor(AfterMergePreprocessingModuleBase):
    def __init__(
        self, source_col_name: str, target_max_col_name: str, target_min_col_name: str, rolling_window: int
    ) -> None:
        self._source_col_name = source_col_name
        self._target_max_col_name = target_max_col_name
        self._target_min_col_name = target_min_col_name
        self._rolling_window = rolling_window

    def __call__(self, result_df: pd.DataFrame) -> pd.DataFrame:

        is_max_series = (
            result_df.groupby("resample_rule")[self._source_col_name]
            .apply(lambda x: x == x.rolling(self._rolling_window).max())
            .rename(self._target_max_col_name)
        )

        is_min_series = (
            result_df.groupby("resample_rule")[self._source_col_name]
            .apply(lambda x: x == x.rolling(self._rolling_window).min())
            .rename(self._target_min_col_name)
        )

        result_df = result_df.reset_index()

        result_df = result_df.merge(is_max_series.reset_index(), on=["resample_rule", "time"], how="left")
        result_df = result_df.merge(is_min_series.reset_index(), on=["resample_rule", "time"], how="left")
        result_df = result_df.set_index("time")

        return result_df
