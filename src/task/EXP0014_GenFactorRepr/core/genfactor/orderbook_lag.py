import pandas as pd

from ...util.const import DFKey
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("orderbook_lag")
class OrderBookLag(GenFactor):
    short_name = "orderbook_lag"
    full_name = "Orderbook Lag"
    description = """
        Measures cumulative order book response lag across specified depths and directional side,
        by comparing current and past order quantities at the best N bid/ask levels.

        - For each orderbook level up to `orderbook_depth`, computes the delta over the `reaction_window`.
        - Supports four modes of aggregation via the `side` parameter:
            - "buy": Sum of lag at buy-side depths.
            - "sell": Sum of lag at sell-side depths.
            - "delta": Difference between buy and sell lag.
            - "abs_sum": Sum of absolute lag on both sides.
    
        This factor captures the net or directional change in visible liquidity over a short horizon,
        which can be useful for diagnosing liquidity asymmetry, replenishment behavior, or reaction to trade flow.
    """

    params = {
        "reaction_window": [3, 5, 10],
        "orderbook_depth": [1, 2, 3, 4, 5],
        "side": ["buy", "sell", "delta", "abs_sum"],
    }

    @property
    def name_with_params(self) -> str:
        return (
            f"{self.short_name}"
            f"_RW{self.selected_params['reaction_window']}"
            f"_OD{self.selected_params['orderbook_depth']}"
            f"_S{self.selected_params['side']}"
        )

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        sp = self.selected_params
        df_tmp = df.copy()

        buy_orderbook_key_list = [
            DFKey.BUY_ORDER_1_QTY_CLOSE,
            DFKey.BUY_ORDER_2_QTY_CLOSE,
            DFKey.BUY_ORDER_3_QTY_CLOSE,
            DFKey.BUY_ORDER_4_QTY_CLOSE,
            DFKey.BUY_ORDER_5_QTY_CLOSE,
        ]
        sell_orderbook_key_list = [
            DFKey.SELL_ORDER_1_QTY_CLOSE,
            DFKey.SELL_ORDER_2_QTY_CLOSE,
            DFKey.SELL_ORDER_3_QTY_CLOSE,
            DFKey.SELL_ORDER_4_QTY_CLOSE,
            DFKey.SELL_ORDER_5_QTY_CLOSE,
        ]

        df_tmp["buy_orderbook_lag"] = 0
        df_tmp["sell_orderbook_lag"] = 0
        for depth in range(sp["orderbook_depth"]):
            buy_orderbook_key = buy_orderbook_key_list[depth]
            sell_orderbook_key = sell_orderbook_key_list[depth]

            df_tmp[f"buy_orderbook_lag"] += df_tmp[buy_orderbook_key] - df_tmp[
                buy_orderbook_key
            ].shift(sp["reaction_window"])

            df_tmp[f"sell_orderbook_lag"] += df_tmp[sell_orderbook_key] - df_tmp[
                sell_orderbook_key
            ].shift(sp["reaction_window"])

        if sp["side"] == "buy":
            df_tmp["orderbook_lag"] = df_tmp["buy_orderbook_lag"]
        elif sp["side"] == "sell":
            df_tmp["orderbook_lag"] = df_tmp["sell_orderbook_lag"]
        elif sp["side"] == "delta":
            df_tmp["orderbook_lag"] = (
                df_tmp["buy_orderbook_lag"] - df_tmp["sell_orderbook_lag"]
            )
        elif sp["side"] == "abs_sum":
            df_tmp["orderbook_lag"] = (
                df_tmp["buy_orderbook_lag"].abs() + df_tmp["sell_orderbook_lag"].abs()
            )
        else:
            raise ValueError(f"Invalid side: {sp['side']}")

        df, col = self._add_to_df(df, df_tmp["orderbook_lag"])
        return df, col
