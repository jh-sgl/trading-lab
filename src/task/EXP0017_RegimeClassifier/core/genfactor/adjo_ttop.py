import numpy as np
import pandas as pd

from ...util.const import DFKey, Num
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("adjo_ttop")
class AdjoTTop(GenFactor):
    short_name = "AdjoTTop"
    full_name = "Adjacent Option-based Tail-Triggered Option Pressure Divergence"
    description = """
        Captures intraday price rejection tails (upper/lower wick strength) and matches them 
        against option-side behavior (volume or price skew), highlighting whether the market is 
        reinforcing or fading those extreme moves.

        Logic:
        - Measure upper and lower wick length as % of total OHLC range.
        - Measure option-side pressure skew (e.g., price or volume tilt).
        - Multiply to get directional rejection confirmation.

        Interpretation:
        - Large negative: long lower wick + put-side surge → fear or protection loading.
        - Large positive: upper wick + call-side demand → breakout reinforcement.
        - Near zero: balanced tails or no confirmation.

        Parameters:
        - level_range: Option strikes to include in skew (1–5).
        - use_volume: Whether to use volume or price skew.
        - ma_window: Optional smoothing.
    """

    params = {
        "level_range": [3, 5],
        "use_volume": [0, 1],  # 0 = use ADJ_*_PRICE, 1 = use ADJ_*_VOL
        "ma_window": [1, 5, 10, 30],
    }

    @property
    def name_with_params(self) -> str:
        p = self.selected_params
        mode = "VOL" if p["use_volume"] else "PRICE"
        return f"{self.short_name}_L{p['level_range']}_{mode}_MA{p['ma_window']}"

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        df_tmp = df.copy()
        L = self.selected_params["level_range"]
        use_vol = self.selected_params["use_volume"]
        MA = self.selected_params["ma_window"]

        open_p = df_tmp[DFKey.FUTURE_PRICE_OPEN]
        close_p = df_tmp[DFKey.FUTURE_PRICE_CLOSE]
        high_p = df_tmp[DFKey.FUTURE_PRICE_HIGH]
        low_p = df_tmp[DFKey.FUTURE_PRICE_LOW]

        range_ = high_p - low_p + Num.EPS
        upper_tail = (high_p - close_p.clip(upper=high_p)) / range_
        lower_tail = (close_p.clip(lower=low_p) - low_p) / range_

        # Positive tail signal if price moves down but wick recovered (bullish rejection)
        tail_score = lower_tail - upper_tail  # ∈ [-1, 1]

        # === Option pressure skew
        key_suffix = "VOL" if use_vol else "PRICE_CLOSE"

        call_keys = [
            getattr(DFKey, f"ADJ_CALL_{i}_{key_suffix}") for i in range(1, L + 1)
        ]
        put_keys = [
            getattr(DFKey, f"ADJ_PUT_{i}_{key_suffix}") for i in range(1, L + 1)
        ]

        call_side = df_tmp[call_keys].sum(axis=1)
        put_side = df_tmp[put_keys].sum(axis=1)

        option_skew = (call_side - put_side) / (call_side + put_side + Num.EPS)

        # === Final signal: tail * option confirmation
        ttop = tail_score * option_skew

        if MA > 1:
            ttop = ttop.rolling(window=MA, min_periods=1).mean()

        df, col = self._add_to_df(df, ttop)
        return df, col
