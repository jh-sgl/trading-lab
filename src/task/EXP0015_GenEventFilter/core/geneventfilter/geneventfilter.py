import logging
import random
from abc import abstractmethod
from typing import Any

import pandas as pd

from ...util.const import DFKey


class GenEventFilter:
    def __init__(self, random_params: bool = False, **kwargs) -> None:
        self.selected_params = {}
        if random_params:
            self.selected_params = {
                key: random.choice(self.params_candidates[key])
                for key in self.params_candidates
            }
        else:
            for key in self.params_candidates:
                if key not in kwargs:
                    raise ValueError(f"Genfactor {self.name} requires {key}")
                self.selected_params[key] = kwargs[key]

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @property
    @abstractmethod
    def params_candidates(self) -> dict[str, list[Any]]:
        pass

    @property
    @abstractmethod
    def name_with_params(self) -> str:
        pass

    @property
    @abstractmethod
    def cols_used(self) -> list[tuple[str, str]]:
        pass

    @abstractmethod
    def _add_event_mask(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        pass

    def add_event_mask(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
        df, event_mask = self._add_event_mask(df)
        df[DFKey.EVENT_MASK] = event_mask

        event_num = df[DFKey.EVENT_MASK].sum()
        total_num = len(df)

        if (ratio := (event_num / total_num)) <= 0.1:
            logging.warn(f"{self.name_with_params} yields very scarce events: {ratio}")

        return df, self.cols_used
