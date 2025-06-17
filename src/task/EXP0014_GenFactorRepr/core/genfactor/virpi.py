import pandas as pd
import numpy as np

from ...util.const import DFKey
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("virpi")
class VIRPI(GenFactor):
    short_name = "VIRPI"
    full_name = "Volatility-Informed Risk Premia Imbalance"
    description = """
        The VIRPI (Volatility-Informed Risk Premia Imbalance) factor captures misalignment
        between market-implied risk premia and theoretical valuation under volatility regimes.
        
        It compares the deviation of actual futures prices from theoretical values
        and normalizes the futures basis against future price (proxy of spot price). This difference is then
        amplified by the VKOSPI z-score to create a volatility-contextualized mispricing signal.

        - High positive values: Overpriced futures in high-volatility regimes → caution or mean reversion.
        - High negative values: Underpriced futures despite high fear → possible reversion/buying opportunity.
        - Near-zero: Market aligned with theory and stable volatility expectations.
    """

    params = {
        "virpi_window": [1, 20, 60],
        "vkospi_lookback": [30, 90, 390],
    }

    @property
    def name_with_params(self) -> str:
        return f"{self.short_name}_VIR{self.selected_params['virpi_window']}_VK{self.selected_params['vkospi_lookback']}"

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        df_tmp = df.copy()

        # 1. Theory deviation
        theory_deviation = (
            df_tmp[DFKey.FUTURE_PRICE_CLOSE] - df_tmp[DFKey.FUTURE_THEORY_CLOSE]
        ) / df_tmp[DFKey.FUTURE_PRICE_CLOSE]

        # 2. Normalized basis
        normalized_basis = df_tmp[DFKey.FUTURE_BASIS_CLOSE] / (
            df_tmp[DFKey.FUTURE_PRICE_CLOSE] + 1e-8
        )  # Spot proxy

        # 3. Risk premia divergence
        premia_divergence = normalized_basis - theory_deviation

        # 4. VKOSPI z-score
        lookback = self.selected_params["vkospi_lookback"]
        vkospi = df_tmp[DFKey.VKOSPI200_REAL_CLOSE]
        vkospi_z = (vkospi - vkospi.rolling(window=lookback, min_periods=1).mean()) / (
            vkospi.rolling(window=lookback, min_periods=1).std() + 1e-8
        )

        # 5. VIRPI raw
        df_tmp["virpi_raw"] = premia_divergence * vkospi_z

        # 6. Smoothing (optional)
        if self.selected_params["virpi_window"] > 1:
            df_tmp["virpi_raw"] = (
                df_tmp["virpi_raw"]
                .rolling(window=self.selected_params["virpi_window"], min_periods=1)
                .mean()
            )

        df, col = self._add_to_df(df, df_tmp["virpi_raw"])
        return df, col
