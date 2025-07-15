import logging
import pandas as pd

from ...util.const import DFKey
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("rivp")
class RIVP(GenFactor):
    short_name = "RIVP"
    full_name = "Regime-Informed Volatility Pivot"
    description = """
        RIVP detects regime shifts by combining volatility regime changes and price breakouts
        from past-confirmed pivot structures. Pivots are confirmed using local extrema with margin.

        - Pivot price is set at confirmation, then held.
        - Pivot type activates after delay to allow price breakout detection.
        - Signal = volatility slope × directional strength × pivot type (lagged).
    """

    params = {
        "vol_window": [5, 10, 20],
        "vol_slope_window": [5, 10],
        "pivot_window": [5, 10, 15],
        "pivot_margin": [2, 3, 5],
        "atr_window": [10, 20],
        "norm_window": [20, 30, 60],
    }

    @property
    def name_with_params(self) -> str:
        sp = self.selected_params
        return (
            f"{self.short_name}_"
            f"VW{sp['vol_window']}_"
            f"VS{sp['vol_slope_window']}_"
            f"PW{sp['pivot_window']}_"
            f"PM{sp['pivot_margin']}_"
            f"AW{sp['atr_window']}_"
            f"NW{sp['norm_window']}"
        )

    @staticmethod
    def detect_pivots(price: pd.Series, w: int, m: int) -> tuple[pd.Series, pd.Series]:
        """
        Detect pivot highs and lows using only past data. Ensures no lookahead bias.

        Parameters:
            price (pd.Series): price series
            w (int): pivot_window (how far back the candidate is)
            m (int): pivot_margin (local extrema check window)

        Returns:
            (pivot_high, pivot_low): boolean series for pivot detection
        """
        candidate = price.shift(w)

        left_max = price.shift(w + 1).rolling(m, min_periods=1).max()
        right_max = price.shift(w - m).rolling(m, min_periods=1).max()

        left_min = price.shift(w + 1).rolling(m, min_periods=1).min()
        right_min = price.shift(w - m).rolling(m, min_periods=1).min()

        pivot_high = (candidate > left_max) & (candidate > right_max)
        pivot_low = (candidate < left_min) & (candidate < right_min)

        return pivot_high, pivot_low

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        df_tmp = df.copy()
        sp = self.selected_params
        price = df_tmp[DFKey.FUTURE_PRICE_CLOSE]

        w = sp["pivot_window"]
        m = sp["pivot_margin"]
        if m >= w:
            adjusted_m = max(1, w - 1)
            logging.warning(
                f"pivot_margin={m} >= pivot_window={w}, auto-adjusting margin to {adjusted_m}."
            )
            m = adjusted_m

        # 1. Volatility slope
        pct_return = price.pct_change()
        rolling_vol = pct_return.rolling(sp["vol_window"], min_periods=1).std()
        vol_slope = rolling_vol.diff(sp["vol_slope_window"])

        # 2. Pivot detection (strictly past data only)
        pivot_high, pivot_low = self.detect_pivots(price, w, m)

        # 3. Confirm pivot price (set only when pivot is detected)
        confirmed_pivot_price = price.where(pivot_high | pivot_low)
        pivot_price = confirmed_pivot_price.ffill()

        # 4. Activate pivot type AFTER it is confirmed (for breakout detection)
        pivot_type_lagged = (pivot_high.astype(int) - pivot_low.astype(int)).shift(w)

        # 5. ATR proxy (for normalization)
        atr_proxy = price.rolling(sp["atr_window"], min_periods=1).apply(
            lambda x: x.max() - x.min(), raw=True
        )

        # 6. Directional strength (distance from confirmed pivot, normalized)
        directional_strength = (price - pivot_price) / (atr_proxy + 1e-8)

        # 7. RIVP raw signal
        rivp_raw = vol_slope * directional_strength * pivot_type_lagged

        # 8. Normalize with robust z-score
        if sp["norm_window"] > 1:
            rivp_smooth = rivp_raw.rolling(sp["norm_window"], min_periods=1).mean()
        else:
            rivp_smooth = rivp_raw

        # 9. Output
        df, col = self._add_to_df(df, rivp_smooth)
        return df, col
