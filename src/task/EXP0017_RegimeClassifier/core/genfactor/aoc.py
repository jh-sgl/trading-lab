import pandas as pd

from ...util.const import DFKey
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("aoc")
class AOC(GenFactor):
    short_name = "AOC"
    full_name = "Adjacent Option Price Convexity"
    description = """
        Measures the curvature (convexity) of the option price structure around the at-the-money level.
        It captures how the prices of adjacent strikes deviate from a linear slope, potentially indicating
        local hedging pressure, skew, or supply/demand kinks.

        Computation:
        - Use adjusted CALL prices at levels 1, 3, and 5 above current FUTURE_PRICE.
        - Fit a simple quadratic function through them (i.e., 2nd difference).
        - Convexity = P_5 - 2 * P_3 + P_1

        Interpretation:
        - Positive: Implied volatility curve is convex (smile)
        - Negative: Inverted convexity (smirk), may reflect dealer pressure or tail hedging

        Parameters:
        - ma_window: Optional moving average to smooth noise.
    """

    params = {
        "ma_window": [1, 5, 10, 30, 60],
    }

    @property
    def name_with_params(self) -> str:
        return f"{self.short_name}_MA{self.selected_params['ma_window']}"

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        df_tmp = df.copy()

        p1 = df_tmp[DFKey.ADJ_CALL_1_PRICE_CLOSE]
        p3 = df_tmp[DFKey.ADJ_CALL_3_PRICE_CLOSE]
        p5 = df_tmp[DFKey.ADJ_CALL_5_PRICE_CLOSE]

        convexity = p5 - 2 * p3 + p1

        if self.selected_params["ma_window"] > 1:
            convexity = convexity.rolling(
                window=self.selected_params["ma_window"], min_periods=1
            ).mean()

        df, col = self._add_to_df(df, convexity)

        return df, col
