import numpy as np
import pandas as pd

from ...util.const import DFKey
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("adjo_aorpa")
class AdjoAORPA(GenFactor):
    short_name = "AdjoAORPA"
    full_name = "Adjacent Option-based Response Asymmetry"
    description = """
        Quantifies how option pricing behavior (across strikes) reacts asymmetrically to 
        short-term price movements, signaling bullish or bearish sentiment reinforcement or fading.

        Logic:
        - Compute price return over short horizon (breakout-like movement).
        - Compute CALL and PUT slope (price change across strike distance).
        - Define asymmetry as:
            - When price moves up, expect CALL slope to increase more.
            - When price moves down, expect PUT slope to steepen more.
        - Final score = directional weight * (CALL_slope - PUT_slope)

        Interpretation:
        - Large positive: bullish breakout with aggressive call repricing.
        - Large negative: sell-off with strong put bid.
        - Weak values: weak option response to movement → possible fade/reversal.

        Parameters:
        - momentum_window: Lookback for price return (%).
        - level_range: Number of option strikes to include.
        - ma_window: Optional smoothing for stability.
    """

    params = {
        "momentum_window": [3, 6, 12],
        "level_range": [3, 5],
        "ma_window": [1, 5, 10, 30, 60],
    }

    @property
    def name_with_params(self) -> str:
        p = self.selected_params
        return f"{self.short_name}_MOM{p['momentum_window']}_L{p['level_range']}_MA{p['ma_window']}"

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        df_tmp = df.copy()
        MOM = self.selected_params["momentum_window"]
        L = self.selected_params["level_range"]
        MA = self.selected_params["ma_window"]

        # === 1. Compute Price Return over Momentum Window
        price_return = df_tmp[DFKey.FUTURE_PRICE_CLOSE].pct_change(periods=MOM)

        # === 2. Option Slope (IV Smile Proxy)
        x = np.arange(1, L + 1).reshape(1, -1)
        x_centered = x - x.mean()
        var_x = (x_centered**2).sum()

        call_price_keys = [
            getattr(DFKey, f"ADJ_CALL_{i}_PRICE_CLOSE") for i in range(1, L + 1)
        ]
        put_price_keys = [
            getattr(DFKey, f"ADJ_PUT_{i}_PRICE_CLOSE") for i in range(1, L + 1)
        ]
        call_prices = np.stack([df_tmp[k] for k in call_price_keys], axis=1)
        put_prices = np.stack([df_tmp[k] for k in put_price_keys], axis=1)

        call_slope = ((x_centered * call_prices).sum(axis=1)) / var_x
        put_slope = ((x_centered * put_prices).sum(axis=1)) / var_x

        # === 3. Asymmetric Response
        sign = np.sign(price_return.fillna(0))  # Direction of move
        slope_diff = call_slope - put_slope  # Which side is pricing more aggressively

        asymmetry_score = (
            sign * slope_diff
        )  # Reinforcement: +1 * CALL_DOMINANT or -1 * PUT_DOMINANT

        if MA > 1:
            asymmetry_score = (
                pd.Series(asymmetry_score, index=df_tmp.index)
                .rolling(window=MA, min_periods=1)
                .mean()
            )

        df, col = self._add_to_df(df, asymmetry_score)
        return df, col
