import numpy as np
import pandas as pd

from ...util.const import DFKey, Num
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("adjo_tscs")
class AdjoTSCS(GenFactor):
    short_name = "AdjoTSCS"
    full_name = "Adjacent Option-based Terminal Skewed Capital Surge"
    description = """
        Measures surge in capital-weighted demand for skewed OTM options into the final hours 
        of the trading session, adjusted by how far each strike is from FUTURE_PRICE.

        Designed for intraday strategies that **must liquidate before close** — focuses on 
        signals that reflect **urgent re-hedging, speculation, or repositioning** into day’s end.

        Logic:
        - Track rolling increase in AMT for far OTM CALLs and PUTs (weighted by distance).
        - Normalize by recent activity to detect abnormal surges.
        - Signed according to futures price direction to reflect directional flow.

        Interpretation:
        - Large positive: bullish price action supported by call-side speculative inflow.
        - Large negative: downside move triggering protective put buying.
        - Zero: No meaningful late-day sentiment shift.

        Parameters:
        - window_delta: How far back to look for AMT change (e.g., past 60 min).
        - level_range: Which ADJ levels to include (e.g., 3~5 = farther OTM only).
        - ma_window: Optional smoothing for model ingestion.
    """

    params = {
        "window_delta": [6, 12, 18],
        "level_range": [3, 5],
        "ma_window": [1, 5, 10, 30],
    }

    @property
    def name_with_params(self) -> str:
        p = self.selected_params
        return f"{self.short_name}_D{p['window_delta']}_L{p['level_range']}_MA{p['ma_window']}"

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        df_tmp = df.copy()
        D = self.selected_params["window_delta"]
        L = self.selected_params["level_range"]
        MA = self.selected_params["ma_window"]

        weights = np.arange(1, L + 1)

        # === Capital flow accumulation (AMT) at farther strikes
        call_amt_now = sum(
            df_tmp[getattr(DFKey, f"ADJ_CALL_{i}_AMT")] * weights[i - 1]
            for i in range(1, L + 1)
        )
        put_amt_now = sum(
            df_tmp[getattr(DFKey, f"ADJ_PUT_{i}_AMT")] * weights[i - 1]
            for i in range(1, L + 1)
        )

        call_amt_past = call_amt_now.shift(D)
        put_amt_past = put_amt_now.shift(D)

        # === Surge in capital positioning
        call_surge = (call_amt_now - call_amt_past).fillna(0)
        put_surge = (put_amt_now - put_amt_past).fillna(0)

        # === Normalize by trailing mean for comparability
        norm_factor = (
            call_amt_now.rolling(D, min_periods=1).mean()
            + put_amt_now.rolling(D, min_periods=1).mean()
            + Num.EPS
        )

        surge_score = (call_surge - put_surge) / norm_factor

        # === Price direction weighting
        price_return = df_tmp[DFKey.FUTURE_PRICE_CLOSE].pct_change(periods=D)
        direction = np.sign(price_return.fillna(0))

        final_score = direction * surge_score

        if MA > 1:
            final_score = final_score.rolling(window=MA, min_periods=1).mean()

        df, col = self._add_to_df(df, final_score)
        return df, col
