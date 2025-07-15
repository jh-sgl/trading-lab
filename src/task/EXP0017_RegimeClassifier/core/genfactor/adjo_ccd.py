import numpy as np
import pandas as pd

from ...util.const import DFKey, Num
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("adjo_ccd")
class AdjoCCD(GenFactor):
    short_name = "AdjoCCD"
    full_name = "Adjacent Option-based Capital Conviction Delta"
    description = """
        Captures how recent changes in capital flow (AMT) into OTM options align with 
        futures price momentum, with strike-weighted aggregation and volatility normalization.

        Enhancements:
        - Strike distance weighting emphasizes speculative positioning.
        - AMT skew dynamics (delta over time) reveal acceleration in capital bias.
        - Volatility-adjusted scaling suppresses noise and emphasizes conviction.

        Parameters:
        - momentum_window: Lookback (bars) for price return and realized vol.
        - level_range: Number of ADJ option strikes to include (1~5).
        - ma_window: Optional smoothing window for final factor.
    """

    params = {
        "momentum_window": [6, 12],  # ~30min to 1hr
        "level_range": [3, 5],
        "ma_window": [1, 5, 10, 30],
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

        # === 1. Price momentum and realized volatility
        price = df_tmp[DFKey.FUTURE_PRICE_CLOSE]
        return_pct = price.pct_change(periods=MOM)
        realized_vol = price.pct_change().rolling(window=MOM, min_periods=1).std()

        # === 2. Strike-weighted AMT aggregation
        weights = np.arange(1, L + 1)

        call_amt = sum(
            df_tmp[getattr(DFKey, f"ADJ_CALL_{i}_AMT")] * weights[i - 1]
            for i in range(1, L + 1)
        )
        put_amt = sum(
            df_tmp[getattr(DFKey, f"ADJ_PUT_{i}_AMT")] * weights[i - 1]
            for i in range(1, L + 1)
        )

        # === 3. Capital skew and its change
        amt_skew = (call_amt - put_amt) / (call_amt + put_amt + Num.EPS)
        skew_delta = amt_skew.diff()

        # === 4. Conviction score: direction × flow change / volatility
        direction = np.sign(return_pct.fillna(0))
        conviction_score = (direction * skew_delta) / (realized_vol + Num.EPS)

        if MA > 1:
            conviction_score = conviction_score.rolling(window=MA, min_periods=1).mean()

        df, col = self._add_to_df(df, conviction_score)
        return df, col
