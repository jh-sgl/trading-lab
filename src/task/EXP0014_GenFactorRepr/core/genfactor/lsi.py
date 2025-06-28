import numpy as np
import pandas as pd

from ...util.const import DFKey
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("lsi")
class LSI(GenFactor):
    short_name = "LSI"
    full_name = "Liquidity Stress Index"
    description = """
        The Liquidity Stress Index (LSI) quantifies the real-time fragility of the order book 
        by combining spread sensitivity, depth imbalance, and persistence of imbalance.

        - Book Tilt Index (BTI): Order book skew favoring either buyers or sellers with distance weighting.
        - Spread Fragility Score (SFS): Measures thinness of the top-of-book liquidity (inverse of depth).
        - Imbalance Persistence (IP): Tracks whether order book skew maintains same direction over time.
        - LSI = |BTI| × SFS × IP (smoothed)

        High LSI indicates a fragile, one-sided market where price is more prone to slipping or jumping.
    """

    params = {
        "depth_levels": [2, 3, 4, 5],
        "ip_window": [3, 5, 10, 20],
        "lsi_window": [1, 5, 10, 20],
    }

    @property
    def name_with_params(self) -> str:
        return f"{self.short_name}_DL{self.selected_params['depth_levels']}_IP{self.selected_params['ip_window']}_LS{self.selected_params['lsi_window']}"

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        df_tmp = df.copy()

        buy_order_dfkeys = [
            DFKey.BUY_ORDER_1_QTY_CLOSE,
            DFKey.BUY_ORDER_2_QTY_CLOSE,
            DFKey.BUY_ORDER_3_QTY_CLOSE,
            DFKey.BUY_ORDER_4_QTY_CLOSE,
            DFKey.BUY_ORDER_5_QTY_CLOSE,
        ]
        sell_order_dfkeys = [
            DFKey.SELL_ORDER_1_QTY_CLOSE,
            DFKey.SELL_ORDER_2_QTY_CLOSE,
            DFKey.SELL_ORDER_3_QTY_CLOSE,
            DFKey.SELL_ORDER_4_QTY_CLOSE,
            DFKey.SELL_ORDER_5_QTY_CLOSE,
        ]

        dlvl = self.selected_params["depth_levels"]

        # --- 1. Book Tilt Index (BTI)
        buy_weighted = sum((i + 1) * df_tmp[buy_order_dfkeys[i]] for i in range(dlvl))
        sell_weighted = sum((i + 1) * df_tmp[sell_order_dfkeys[i]] for i in range(dlvl))
        bti_num = buy_weighted - sell_weighted
        bti_denom = buy_weighted + sell_weighted + 1e-8
        df_tmp["bti"] = bti_num / bti_denom

        # --- 2. Spread Fragility Score (SFS)
        best_bid_volume = df_tmp[buy_order_dfkeys[0]] + 1e-8
        best_ask_volume = df_tmp[sell_order_dfkeys[0]] + 1e-8
        sfs_bid = 1.0 / best_bid_volume
        sfs_ask = 1.0 / best_ask_volume

        # Directional SFS based on BTI sign
        df_tmp["sfs"] = sfs_bid.where(df_tmp["bti"] > 0, sfs_ask)

        # --- 3. Imbalance Persistence (IP)
        df_tmp["bti_sign"] = np.sign(df_tmp["bti"])
        df_tmp["ip"] = (
            df_tmp["bti_sign"]
            .rolling(window=self.selected_params["ip_window"], min_periods=1)
            .mean()
        )

        # --- 4. LSI Raw
        df_tmp["lsi_raw"] = df_tmp["sfs"] * df_tmp["ip"] * df_tmp["bti"].abs()

        if self.selected_params["lsi_window"] > 1:
            df_tmp["lsi_raw"] = (
                df_tmp["lsi_raw"]
                .rolling(window=self.selected_params["lsi_window"], min_periods=1)
                .mean()
            )

        df, col = self._add_to_df(df, df_tmp["lsi_raw"])
        return df, col
