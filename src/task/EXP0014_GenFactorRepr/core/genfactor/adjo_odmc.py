import numpy as np
import pandas as pd

from ...util.const import DFKey
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("adjo_odmc")
class AdjoODMC(GenFactor):
    short_name = "AdjoODMC"
    full_name = "Adjacent Option-Driven Momentum Confirmation"
    description = """
        Measures alignment between recent futures price momentum and adjacent option market signals.

        Logic:
        - Computes short-term futures momentum.
        - Computes net demand skew in nearby options (CALL vs PUT volume tilt).
        - Returns their product → high when both point in the same direction, negative when diverging.

        Interpretation:
        - Strong positive: Bullish price move supported by call-side demand.
        - Strong negative: Bearish move supported by put-side interest.
        - Near zero or negative alignment: Momentum may lack conviction from option flow.

        Parameters:
        - momentum_window: Lookback (in 5-min bars) to compute futures momentum.
        - level_range: Number of ADJ option levels to include.
        - ma_window: Optional smoothing of the final output.
    """

    params = {
        "momentum_window": [3, 6, 12],  # e.g., past 15, 30, 60 minutes
        "level_range": [1, 3, 5],
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

        # === 1. Price Momentum (FUTURE_PRICE_CLOSE momentum over past N bars)
        momentum = df_tmp[DFKey.FUTURE_PRICE_CLOSE].pct_change(periods=MOM)

        # === 2. Option Volume Tilt
        call_keys = [getattr(DFKey, f"ADJ_CALL_{i}_VOLUME") for i in range(1, L + 1)]
        put_keys = [getattr(DFKey, f"ADJ_PUT_{i}_VOLUME") for i in range(1, L + 1)]
        call_vol = df_tmp[call_keys].sum(axis=1)
        put_vol = df_tmp[put_keys].sum(axis=1)

        volume_tilt = (call_vol - put_vol) / (call_vol + put_vol + 1e-8)

        # === 3. Alignment Score
        alignment = momentum * volume_tilt

        if MA > 1:
            alignment = alignment.rolling(window=MA, min_periods=1).mean()

        df, col = self._add_to_df(df, alignment)
        return df, col
