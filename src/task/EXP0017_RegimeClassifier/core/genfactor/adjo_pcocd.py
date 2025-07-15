import numpy as np
import pandas as pd

from ...util.const import DFKey, Num
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("adjo_pcocd")
class AdjoPCOCD(GenFactor):
    short_name = "AdjoPCOCD"
    full_name = "Put-Call OI Curve Difference (Convexity)"
    description = """
        Measures the curvature (second-difference) of open interest across adjacent strikes,
        comparing convexity between puts and calls. Indicates where OI is concentrated.

        - Hump-shaped curve (ATM heavy): positive curvature.
        - U-shaped curve (wing heavy): negative curvature.
        - Skew in curvature across puts and calls reflects asymmetric exposure or hedging.

        Parameters:
        - ma_window: Optional smoothing of the resulting signal.
    """

    params = {
        "ma_window": [1, 5, 10],
    }

    @property
    def name_with_params(self) -> str:
        return f"{self.short_name}_MA{self.selected_params['ma_window']}"

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        df_tmp = df.copy()

        call_curve = df_tmp[DFKey.ADJ_CALL_3_OPENINT_CLOSE] - 0.5 * (
            df_tmp[DFKey.ADJ_CALL_2_OPENINT_CLOSE]
            + df_tmp[DFKey.ADJ_CALL_4_OPENINT_CLOSE]
        )

        put_curve = df_tmp[DFKey.ADJ_PUT_3_OPENINT_CLOSE] - 0.5 * (
            df_tmp[DFKey.ADJ_PUT_2_OPENINT_CLOSE]
            + df_tmp[DFKey.ADJ_PUT_4_OPENINT_CLOSE]
        )

        curve_diff = call_curve - put_curve

        if self.selected_params["ma_window"] > 1:
            curve_diff = curve_diff.rolling(
                window=self.selected_params["ma_window"], min_periods=1
            ).mean()

        df, col = self._add_to_df(df, curve_diff)
        return df, col
