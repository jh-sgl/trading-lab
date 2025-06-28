import numpy as np
import pandas as pd

from ...util.const import DFKey
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("hsfi")
class HSFI(GenFactor):
    short_name = "HSFI"
    full_name = "HFT Shadow Flow Index"
    description = """
        The HFT Shadow Flow Index (HSFI) detects latent HFT-like pressure by measuring
        order book instability, reversal traps, and synthetic liquidity.
        It captures the divergence between visible order flow pressure and realized stability.

        - BFR: Book Flicker Rate, detects transient liquidity.
        - IRD: Imbalance Reversal Divergence, penalizes order book tilt that fails to realize.
        - TTMR: Tick-Time Micro Reversal, frequent traps imply adverse selection.

        HSFI is useful for identifying deceptive liquidity environments where price is unlikely
        to respond to visible imbalance due to aggressive liquidity recycling or HFT absorption.
    """

    params = {
        "hsfi_window": [1, 5, 10, 15],
        "flicker_window": [2, 3, 5],
        "bfr_weight": [0.3, 0.5, 0.7],
        "ird_weight": [0.3, 0.5, 0.7],
        "ttmr_weight": [0.3, 0.5, 0.7],
    }

    @property
    def name_with_params(self) -> str:
        return (
            f"{self.short_name}_HS{self.selected_params['hsfi_window']}_"
            f"FL{self.selected_params['flicker_window']}_"
            f"BFR{int(self.selected_params['bfr_weight'] * 10)}_"
            f"IRD{int(self.selected_params['ird_weight'] * 10)}_"
            f"TTMR{int(self.selected_params['ttmr_weight'] * 10)}"
        )

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        df_tmp = df.copy()

        # Book Flicker Rate: instability in top-of-book liquidity
        buy_depth = df_tmp[DFKey.BUY_ORDER_1_QTY_CLOSE]
        sell_depth = df_tmp[DFKey.SELL_ORDER_1_QTY_CLOSE]
        flicker = (
            (buy_depth.pct_change().abs() + sell_depth.pct_change().abs())
            .rolling(window=self.selected_params["flicker_window"], min_periods=1)
            .mean()
        )
        bfr = flicker.fillna(0)

        # Imbalance Reversal Divergence: pressure not leading to price moves
        imbalance = (buy_depth - sell_depth) / (buy_depth + sell_depth + 1e-6)
        price_change = df_tmp[DFKey.FUTURE_PRICE_CLOSE].diff().fillna(0)
        imbalance_sign = imbalance.shift(1).fillna(0).apply(np.sign)
        price_sign = price_change.apply(np.sign)
        mismatch = (imbalance_sign != price_sign).astype(int)
        ird = mismatch.rolling(5).mean().fillna(0)

        # Tick-Time Micro Reversal: noisy, back-and-forth microtraps
        tick_move = price_sign.diff().abs()
        ttmr = tick_move.rolling(5).sum().fillna(0) / 5

        # Combine components
        hsfi_raw = (
            self.selected_params["bfr_weight"] * bfr
            + self.selected_params["ird_weight"] * ird
            + self.selected_params["ttmr_weight"] * ttmr
        )

        if self.selected_params["hsfi_window"] > 1:
            hsfi_raw = hsfi_raw.rolling(
                window=self.selected_params["hsfi_window"], min_periods=1
            ).mean()

        df, col = self._add_to_df(df, hsfi_raw)

        return df, col
