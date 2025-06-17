import logging
import random
from typing import Any

import numpy as np
import pandas as pd

from ...util.const import DFKey
from ...util.registry import register_genfactor


class GenFactor:
    short_name: str
    full_name: str
    description: str
    params: dict[str, list[Any]]
    selected_params: dict[str, Any]

    def __init__(self, **kwargs) -> None:
        self.selected_params = {}
        for key in self.params:
            if key not in kwargs:
                self.selected_params[key] = random.choice(self.params[key])
            else:
                self.selected_params[key] = kwargs[key]

    @property
    def name_with_params(self) -> str:
        raise NotImplementedError("Subclass should implement this function")

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        raise NotImplementedError("Subclass should implement this function")

    def _add_to_df(
        self, df: pd.DataFrame, genfactor: pd.Series
    ) -> tuple[pd.DataFrame, tuple[str, str]]:
        col = ("genfactor", self.name_with_params)
        df[col] = genfactor
        if df[col].isna().sum() > 1000:
            logging.warning(f"Genfactor {col} has {df[col].isna().sum()} NaNs")
        return df, col
