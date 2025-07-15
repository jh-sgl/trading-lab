import numpy as np
import pandas as pd

from ...util.const import DFKey
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("ttmr")
class TTMR(GenFactor):
    short_name = "TTMR"
    full_name = "Tick-Time Micro Reversal"
    description = """
        Tick-Time Micro Reversal (TTMR) quantifies microstructure noise by detecting frequent
        reversals in tick-level price movements. High TTMR suggests adverse selection,
        liquidity traps, or aggressive flow being absorbed at the micro level.
    """

    params = {"ttmr_window": [3, 5, 10, 20, 30]}

    @property
    def name_with_params(self) -> str:
        return f"{self.short_name}_W{self.selected_params['ttmr_window']}"

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        df_tmp = df.copy()
        price_change = df_tmp[DFKey.FUTURE_PRICE_CLOSE].diff().fillna(0)
        price_sign = price_change.apply(np.sign)
        tick_move = price_sign.diff().abs()

        ttmr = (
            tick_move.rolling(self.selected_params["ttmr_window"], min_periods=1).sum()
            / self.selected_params["ttmr_window"]
        ).fillna(0)

        df, col = self._add_to_df(df, ttmr)
        return df, col
