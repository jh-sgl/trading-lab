import pandas as pd

from ...util.const import DFKey
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("putcall_iv_skew_slope")
class PutCallIVSkewSlope(GenFactor):
    short_name = "PutCallIVSkewSlope"
    full_name = "Put-Call Implied Volatility Skew Slope"
    description = """
        Calculates the implied volatility skew slope, capturing sentiment bias between calls and puts.

        The skew slope is determined by:
        - Computing the implied volatility (IV) slope of calls and puts separately,
          based on specified strike distances from the at-the-money (ATM) options.
        - Measuring the difference between call skew and put skew to reflect the directional sentiment.

        This metric helps identify:
        - Market expectations of upward versus downward price movements.
        - Demand pressure on out-of-the-money (OTM) options, indicating speculative or hedging activities.
    """

    params = {
        "ma_window": [1, 5, 10, 25, 50],
    }

    @property
    def name_with_params(self) -> str:
        return f"{self.short_name}_MA{self.selected_params['ma_window']}"

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        df_tmp = df.copy()

        call_skew = (
            df_tmp[DFKey.CALL_IV_2ND_UP_CLOSE] - df_tmp[DFKey.CALL_IV_CLOSE]
        ) / 5
        put_skew = (
            df_tmp[DFKey.PUT_IV_CLOSE] - df_tmp[DFKey.PUT_IV_2ND_DOWN_CLOSE]
        ) / 5
        skew_slope = call_skew - put_skew

        if self.selected_params["ma_window"] > 1:
            skew_slope = skew_slope.rolling(
                window=self.selected_params["ma_window"], min_periods=1
            ).mean()

        df, col = self._add_to_df(df, skew_slope)

        return df, col
