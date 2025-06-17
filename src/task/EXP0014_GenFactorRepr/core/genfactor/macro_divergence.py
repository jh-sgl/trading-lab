import numpy as np
import pandas as pd

from ...util.const import DFKey
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("macro_divergence")
class MacroDivergenceFactor(GenFactor):
    short_name = "macro_divergence"
    full_name = "Cross-Asset Macro Divergence Signal"
    description = """
        Calculates a macro divergence signal based on bond spread and USD futures movement
        over a specified rolling window. Intended to capture cross-asset tension.
    """

    params = {
        "macro_component": ["bond_spread", "usd_delta", "bond_spread+usd_delta"],
        "macro_window": [5, 10, 20, 40, 60, 120],
    }

    @property
    def name_with_params(self) -> str:
        return f"{self.short_name}_{self.selected_params['macro_component']}_W{self.selected_params['macro_window']}"

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        df_tmp = df.copy()
        macro_component = self.selected_params["macro_component"]
        window = self.selected_params["macro_window"]
        macro_signal = pd.Series(0, index=df_tmp.index)

        if "bond_spread" in macro_component:
            df_tmp["bond_spread"] = (
                df_tmp[DFKey.BOND_167_CLOSE] - df_tmp[DFKey.BOND_165_CLOSE]
            )
            macro_signal += df_tmp["bond_spread"].diff(window)

        if "usd_delta" in macro_component:
            macro_signal += df_tmp[DFKey.USD_PRICE_CLOSE].diff(window)

        df, col = self._add_to_df(df, macro_signal.rename("macro_signal"))
        return df, col
