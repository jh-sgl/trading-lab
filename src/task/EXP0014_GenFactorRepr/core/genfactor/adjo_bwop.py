import numpy as np
import pandas as pd

from ...util.const import DFKey, Num
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("adjo_bwop")
class AdjoBWOP(GenFactor):
    short_name = "AdjoBWOP"
    full_name = "Adjacent Option-based Breakout-Weighted Option Pressure"
    description = """
        Detects whether intraday breakouts (based on OHLC range) are supported by option-side capital pressure.

        Combines:
        - Range breakout intensity: (CLOSE - OPEN) / (HIGH - LOW)
        - Option pressure imbalance: OTM CALL vs PUT AMT skew
        - Final output reflects whether directional move is reinforced or fading

        Interpretation:
        - Large positive: bullish breakout backed by aggressive call positioning.
        - Large negative: bearish range move reinforced by put capital flow.
        - Near zero: indecisive or unconfirmed moves.

        Parameters:
        - level_range: How many OTM levels to include in ADJ AMT skew.
        - ma_window: Optional smoothing of the final signal.
    """

    params = {
        "level_range": [3, 5],
        "ma_window": [1, 5, 10, 30],
    }

    @property
    def name_with_params(self) -> str:
        p = self.selected_params
        return f"{self.short_name}_L{p['level_range']}_MA{p['ma_window']}"

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        df_tmp = df.copy()
        L = self.selected_params["level_range"]
        MA = self.selected_params["ma_window"]

        # === 1. Price breakout score
        open_p = df_tmp[DFKey.FUTURE_PRICE_OPEN]
        close_p = df_tmp[DFKey.FUTURE_PRICE_CLOSE]
        high_p = df_tmp[DFKey.FUTURE_PRICE_HIGH]
        low_p = df_tmp[DFKey.FUTURE_PRICE_LOW]

        range_ = high_p - low_p
        breakout_score = (close_p - open_p) / (range_ + Num.EPS)  # normalized [-1, 1]

        # === 2. Option AMT skew (capital pressure from options)
        call_amt_keys = [getattr(DFKey, f"ADJ_CALL_{i}_AMT") for i in range(1, L + 1)]
        put_amt_keys = [getattr(DFKey, f"ADJ_PUT_{i}_AMT") for i in range(1, L + 1)]

        call_amt = df_tmp[call_amt_keys].sum(axis=1)
        put_amt = df_tmp[put_amt_keys].sum(axis=1)

        amt_skew = (call_amt - put_amt) / (call_amt + put_amt + Num.EPS)

        # === 3. Final output: directional price move × capital support
        bwop = breakout_score * amt_skew

        if MA > 1:
            bwop = bwop.rolling(window=MA, min_periods=1).mean()

        df, col = self._add_to_df(df, bwop)
        return df, col
