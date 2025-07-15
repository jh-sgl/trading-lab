import pandas as pd

from ...util.const import DFKey
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("straddleimbalance")
class StraddleImbalance(GenFactor):
    short_name = "StraddleImb"
    full_name = "Synthetic Straddle Imbalance"
    description = """
        Estimates the net bias embedded in synthetic ATM straddles, comparing the price of CALL and PUT options 
        at the 1st adjacent level above and below the current futures price.

        Motivation:
        - In an efficient market with no skew or bias, a symmetric call and put should have similar value.
        - An imbalance may indicate directional skew, implied drift, or asymmetric hedging pressure.

        Computation:
        - Use: CALL_1_PRICE and PUT_1_PRICE (both CLOSE)
        - Imbalance = (CALL_1_PRICE - PUT_1_PRICE) / (CALL_1_PRICE + PUT_1_PRICE + ε)

        Interpretation:
        - Positive: Call premium > Put → bullish lean
        - Negative: Put premium > Call → bearish lean
        - Close to 0: Balanced market, neutral straddle

        Parameters:
        - ma_window: Optional smoothing window to reduce noise.
    """

    params = {
        "ma_window": [1, 5, 10, 30, 60],
    }

    @property
    def name_with_params(self) -> str:
        return f"{self.short_name}_MA{self.selected_params['ma_window']}"

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        df_tmp = df.copy()
        eps = 1e-8

        call_price = df_tmp[DFKey.ADJ_CALL_1_PRICE_CLOSE]
        put_price = df_tmp[DFKey.ADJ_PUT_1_PRICE_CLOSE]

        straddle_imb = (call_price - put_price) / (call_price + put_price + eps)

        if self.selected_params["ma_window"] > 1:
            straddle_imb = straddle_imb.rolling(
                window=self.selected_params["ma_window"], min_periods=1
            ).mean()

        df, col = self._add_to_df(df, straddle_imb)

        return df, col
