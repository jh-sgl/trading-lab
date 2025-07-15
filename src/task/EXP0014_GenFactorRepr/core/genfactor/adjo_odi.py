import numpy as np
import pandas as pd

from ...util.const import DFKey, Num
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("adjo_odi")
class AdjoODI(GenFactor):
    short_name = "AdjoODI"
    full_name = "Net Directional Open Interest Imbalance"
    description = """
        Measures net imbalance between open interest of call options above and put options below the futures price.
        Captures directional skew or inventory hedging pressure.

        - Positive: More call OI above than put OI below = bullish skew
        - Negative: More put OI = bearish hedging/speculation

        Parameters:
        - ma_window: Smoothing window (in bars) [default=1]
        - norm_total_oi: Normalize by total OI to scale relative size
        - level_range: How many strikes above/below to include (1~5)
    """

    params = {
        "ma_window": [1, 5, 10],
        "norm_total_oi": [False, True],
        "level_range": [1, 2, 3, 4, 5],
    }

    @property
    def name_with_params(self) -> str:
        return (
            f"{self.short_name}_MA{self.selected_params['ma_window']}"
            f"_N{int(self.selected_params['norm_total_oi'])}"
            f"_L{self.selected_params['level_range']}"
        )

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        df_tmp = df.copy()
        level_range = self.selected_params["level_range"]

        # Aggregate over adjustable levels
        call_oi = sum(
            df_tmp[getattr(DFKey, f"ADJ_CALL_{i}_OPENINT_CLOSE")]
            for i in range(1, level_range + 1)
        )
        put_oi = sum(
            df_tmp[getattr(DFKey, f"ADJ_PUT_{i}_OPENINT_CLOSE")]
            for i in range(1, level_range + 1)
        )

        imbalance = call_oi - put_oi

        if self.selected_params["norm_total_oi"]:
            total_oi = call_oi + put_oi
            imbalance = imbalance / (total_oi + 1e-8)  # prevent division by zero

        if self.selected_params["ma_window"] > 1:
            imbalance = imbalance.rolling(
                window=self.selected_params["ma_window"], min_periods=1
            ).mean()

        df, col = self._add_to_df(df, imbalance)
        return df, col
