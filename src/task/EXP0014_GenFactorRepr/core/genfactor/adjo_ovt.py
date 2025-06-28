import numpy as np
import pandas as pd

from ...util.const import DFKey
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("adjo_ovt")
class AdjoOVT(GenFactor):
    short_name = "AdjoOVT"
    full_name = "Adjacent-option-based Directional Option Volume Tilt"
    description = """
        Measures volume-based directional skew in adjacent options by comparing 
        total traded volume in OTM CALLs versus OTM PUTs near the current FUTURE_PRICE.

        The feature estimates which side of the market (calls vs puts) is receiving more attention
        from participants — potentially signaling bullish or bearish intra-day bias.

        Parameters:
        - level_range: Number of strike levels to include (1–5).
        - ma_window: Optional smoothing window to reduce noise.

        Interpretation:
        - Positive value: More volume in OTM CALLs → potential upside sentiment.
        - Negative value: More volume in OTM PUTs → downside hedging or bearish bias.
    """

    params = {
        "level_range": [1, 3, 5],
        "ma_window": [1, 5, 10, 30, 60],
    }

    @property
    def name_with_params(self) -> str:
        return f"{self.short_name}_L{self.selected_params['level_range']}_MA{self.selected_params['ma_window']}"

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        df_tmp = df.copy()
        level_range = self.selected_params["level_range"]
        ma_window = self.selected_params["ma_window"]

        # Collect volume over selected adjacent strikes
        call_keys = [
            getattr(DFKey, f"ADJ_CALL_{i}_VOLUME") for i in range(1, level_range + 1)
        ]
        put_keys = [
            getattr(DFKey, f"ADJ_PUT_{i}_VOLUME") for i in range(1, level_range + 1)
        ]

        call_vol = df_tmp[call_keys].sum(axis=1)
        put_vol = df_tmp[put_keys].sum(axis=1)

        tilt = (call_vol - put_vol) / (
            call_vol + put_vol + 1e-8
        )  # normalized tilt, avoids div-by-zero

        if ma_window > 1:
            tilt = tilt.rolling(window=ma_window, min_periods=1).mean()

        df, col = self._add_to_df(df, tilt)
        return df, col
