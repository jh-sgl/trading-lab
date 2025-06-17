import pandas as pd

from ...util.const import DFKey
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("bsfs")
class BookSpreadFragilityScore(GenFactor):
    short_name = "bsfs"
    full_name = "Book Spread Fragility Score"
    description = """
        The Book Spread Fragility Score (BSFS) measures the vulnerability of the order book 
        near the spread by examining the inverse of the volume at the best bid and ask levels. 
        It provides insight into how easily prices can move due to limited liquidity.

        Parameter `side` determines the mode of calculation:
        - "bid": Focuses on bid-side fragility (1 / best bid volume).
        - "ask": Focuses on ask-side fragility (1 / best ask volume).
        - "delta": Measures directional imbalance (bid fragility - ask fragility).
        - "abs_sum": Measures total top-level fragility (|1/bid| + |1/ask|).

        Higher values in BSFS indicate thinner liquidity and a more fragile top-of-book environment,
        suggesting potential for larger slippage under aggressive trading.
    """

    params = {
        "depth_levels": [1, 2, 3, 4, 5],
        "side": ["bid", "ask", "delta", "abs_sum"],
    }

    @property
    def name_with_params(self) -> str:
        return f"{self.short_name}_DL{self.selected_params['depth_levels']}_S{self.selected_params['side']}"

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        df_tmp = df.copy()

        buy_order_keys = [
            DFKey.BUY_ORDER_1_CLOSE,
            DFKey.BUY_ORDER_2_CLOSE,
            DFKey.BUY_ORDER_3_CLOSE,
            DFKey.BUY_ORDER_4_CLOSE,
            DFKey.BUY_ORDER_5_CLOSE,
        ]
        sell_order_keys = [
            DFKey.SELL_ORDER_1_CLOSE,
            DFKey.SELL_ORDER_2_CLOSE,
            DFKey.SELL_ORDER_3_CLOSE,
            DFKey.SELL_ORDER_4_CLOSE,
            DFKey.SELL_ORDER_5_CLOSE,
        ]

        buy_volume = pd.Series(0, index=df_tmp.index)
        sell_volume = pd.Series(0, index=df_tmp.index)
        for depth in range(self.selected_params["depth_levels"]):
            buy_volume += df_tmp[buy_order_keys[depth]]
            sell_volume += df_tmp[sell_order_keys[depth]]

        sfs_bid = 1.0 / (buy_volume + 1e-8)
        sfs_ask = 1.0 / (sell_volume + 1e-8)

        if self.selected_params["side"] == "bid":
            sfs = sfs_bid
        elif self.selected_params["side"] == "ask":
            sfs = sfs_ask
        elif self.selected_params["side"] == "delta":
            sfs = sfs_bid - sfs_ask
        elif self.selected_params["side"] == "abs_sum":
            sfs = sfs_bid.abs() + sfs_ask.abs()

        return self._add_to_df(df, sfs)
