import numpy as np
import pandas as pd

from ...util.const import DFKey, Num
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("adjo_pcos")
class AdjoPCOS(GenFactor):
    short_name = "AdjoPCOS"
    full_name = "Put-Call OI Symmetry"
    description = """
        Measures the degree of symmetry between open interest levels across equidistant call and put strikes.
        Asymmetry indicates directional skew, hedging imbalance, or speculative crowding.

        - Low symmetry score = balanced positioning.
        - High score = asymmetrical exposure (e.g., excess downside protection).
        
        Parameters:
        - levels: Number of adjacent strikes to include (1 = closest ATM, up to 5).
        - ma_window: Optional smoothing of the symmetry score.
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

        diffs = []
        for i in range(1, L + 1):
            call_key = getattr(DFKey, f"ADJ_CALL_{i}_OPENINT_CLOSE")
            put_key = getattr(DFKey, f"ADJ_PUT_{i}_OPENINT_CLOSE")
            diffs.append((df_tmp[call_key] - df_tmp[put_key]).abs())

        symmetry_score = sum(diffs) / L

        if self.selected_params["ma_window"] > 1:
            symmetry_score = symmetry_score.rolling(
                window=self.selected_params["ma_window"], min_periods=1
            ).mean()

        df, col = self._add_to_df(df, symmetry_score)
        return df, col
