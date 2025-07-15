import logging

import pandas as pd

from ...util.const import DFKey
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("vkospi_zscore")
class VKOSPIZScore(GenFactor):
    short_name = "vkospi_zscore"
    full_name = "VKOSPI Z-Score"
    description = """
        The VKOSPIZScore factor calculates the standardized z-score of VKOSPI over a rolling window. 
        It provides a volatility regime signal used to contextualize market sentiment.

        - High values: Elevated volatility (risk-off/fear).
        - Low/negative values: Complacent or low-volatility regime.
        - Near-zero: Neutral volatility state.
    """

    params = {"ma_window": [30, 90, 120, 390, 1200]}

    @property
    def name_with_params(self) -> str:
        return f"{self.short_name}_L{self.selected_params['ma_window']}"

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        ma_window = self.selected_params["ma_window"]
        df_tmp = df.copy()

        vkospi = df_tmp[DFKey.VKOSPI200_REAL_CLOSE]
        vkospi_z = (vkospi - vkospi.rolling(window=ma_window, min_periods=1).mean()) / (
            vkospi.rolling(window=ma_window, min_periods=1).std() + 1e-8
        )

        df, col = self._add_to_df(df, vkospi_z)
        return df, col
