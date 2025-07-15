import numpy as np
import pandas as pd

from ...util.const import DFKey, Num
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("adjo_pcods")
class AdjoPCODS(GenFactor):
    short_name = "AdjoPCODS"
    full_name = "Put-Call OI Dispersion Skew"
    description = """
        Measures asymmetry in how dispersed open interest is across adjacent call and put strikes.
        Dispersion is calculated using standard deviation across a configurable number of OTM levels.

        - Positive skew: calls have broader OI distribution — speculative upside.
        - Negative skew: puts are more distributed — hedging or tail exposure.
        - Symmetrical = near-ATM focus.

        Parameters:
        - levels: Number of adjacent strikes to include (1 = closest ATM, 5 = farthest OTM).
        - ma_window: Smoothing window over time.
    """

    params = {
        "levels": [3, 4, 5],
        "ma_window": [1, 5, 10],
    }

    @property
    def name_with_params(self) -> str:
        return f"{self.short_name}_L{self.selected_params['levels']}_MA{self.selected_params['ma_window']}"

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        df_tmp = df.copy()
        L = self.selected_params["levels"]

        call_keys = [
            getattr(DFKey, f"ADJ_CALL_{i}_OPENINT_CLOSE") for i in range(1, L + 1)
        ]
        put_keys = [
            getattr(DFKey, f"ADJ_PUT_{i}_OPENINT_CLOSE") for i in range(1, L + 1)
        ]

        call_disp = df_tmp[call_keys].std(axis=1)
        put_disp = df_tmp[put_keys].std(axis=1)

        skew = call_disp - put_disp

        if self.selected_params["ma_window"] > 1:
            skew = skew.rolling(
                window=self.selected_params["ma_window"], min_periods=1
            ).mean()

        df, col = self._add_to_df(df, skew)
        return df, col
