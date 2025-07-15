import numpy as np
import pandas as pd

from ...util.const import DFKey
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("bti")
class BookTiltIndex(GenFactor):
    short_name = "BTI"
    full_name = "Book Tilt Index"
    description = """
        The Book Tilt Index (BTI) quantifies the directional skew of the order book depth, 
        using distance-weighted order volumes. A positive value indicates buy-side dominance,
        while a negative value implies sell-side dominance.
    """

    params = {
        "depth_levels": [2, 3, 4, 5],
    }

    @property
    def name_with_params(self) -> str:
        return f"{self.short_name}_DL{self.selected_params['depth_levels']}"

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        df_tmp = df.copy()
        dlvl = self.selected_params["depth_levels"]

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

        buy_weighted = sum((i + 1) * df_tmp[buy_order_dfkeys[i]] for i in range(dlvl))
        sell_weighted = sum((i + 1) * df_tmp[sell_order_dfkeys[i]] for i in range(dlvl))
        bti_num = buy_weighted - sell_weighted
        bti_denom = buy_weighted + sell_weighted + 1e-8
        df_tmp["bti"] = bti_num / bti_denom

        return self._add_to_df(df, df_tmp["bti"])
