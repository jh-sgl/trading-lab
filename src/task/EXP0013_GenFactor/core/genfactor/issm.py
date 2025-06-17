import numpy as np
import pandas as pd

from ...util.const import DFKey
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("issm")
class ISSM(GenFactor):
    short_name = "ISSM"
    full_name = "Implied Sentiment Spread Momentum"
    description = """
        ISSM captures the divergence between directional futures momentum and 
        skewed sentiment in the options market, measured by changes in implied volatility skew.

        - Computes short-term price momentum from K200 futures.
        - Computes skew as normalized difference between call and put implied vols.
        - ISSM = sign of futures momentum × change in IV skew.
        - Positive ISSM: price rise with increasing call-side dominance.
        - Negative ISSM: price rise with increasing put-side protection.
    """

    params = {
        "momentum_window": [1, 2, 3, 5, 10],
        "skew_smooth_window": [3, 5, 10],
        "norm_window": [20, 60],
    }

    @property
    def name_with_params(self) -> str:
        sp = self.selected_params
        return (
            f"{self.short_name}_"
            f"MW{sp['momentum_window']}_"
            f"SW{sp['skew_smooth_window']}_"
            f"NW{sp['norm_window']}"
        )

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        df_tmp = df.copy()
        sp = self.selected_params

        price = df_tmp[DFKey.FUTURE_PRICE_CLOSE]
        call_iv = df_tmp[DFKey.CALL_IV_CLOSE]
        put_iv = df_tmp[DFKey.PUT_IV_CLOSE]

        # 1. Futures short-term momentum
        momentum = price.diff(sp["momentum_window"]).apply(np.sign)

        # 2. IV Skew (normalized difference)
        iv_skew = (call_iv - put_iv) / (call_iv + put_iv + 1e-8)

        # 3. Change in skew (delta skew)
        skew_smooth = iv_skew.rolling(sp["skew_smooth_window"], min_periods=1).mean()
        skew_change = skew_smooth.diff()

        # 4. Raw ISSM signal
        issm_raw = momentum * skew_change

        # 5. Normalize (robust z-score or smooth mean)
        if sp["norm_window"] > 1:
            issm_smooth = issm_raw.rolling(sp["norm_window"], min_periods=1).mean()
        else:
            issm_smooth = issm_raw

        # 6. Output
        df, col = self._add_to_df(df, issm_smooth)
        return df, col
