import numpy as np
import pandas as pd

from ...util.const import DFKey
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("bond_10y_shock")
class Bond10YShock(GenFactor):
    short_name = "bond_10y_shock"
    full_name = "10Y KTB Z-Score Shock"
    description = """
        Computes z-scored log-return shocks for the 3-Year Korean Treasury Bond.
        Captures macroeconomic signal via short-term interest rate movements.
    """

    params = {
        "shock_window": [5, 10, 20, 40, 60, 120],
    }

    @property
    def name_with_params(self) -> str:
        sw = self.selected_params["shock_window"]
        return f"{self.short_name}_SW{sw}"

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        sw = self.selected_params["shock_window"]
        log_ret = np.log(df[DFKey.BOND_167_CLOSE] + 1e-8).diff()
        zshock = (log_ret - log_ret.rolling(sw, min_periods=1).mean()) / (
            log_ret.rolling(sw, min_periods=1).std() + 1e-8
        )
        df, col = self._add_to_df(df, zshock)
        return df, col
