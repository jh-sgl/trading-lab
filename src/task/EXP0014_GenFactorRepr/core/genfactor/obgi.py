import pandas as pd

from ...util.const import DFKey
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("obgi")
class OBGI(GenFactor):
    short_name = "OBGI"
    full_name = "Order Book Gradient Imbalance"
    description = """
        OBGI captures latent directional pressure in the order book by computing the gradient imbalance
        between bid and ask sides, weighted by their price distance to the mid-price.

        - Uses 5-level order book data for buy/sell side
        - Measures pressure asymmetry across the book depth
        - Positive values indicate net bullish pressure; negative indicates bearish
    """

    params = {
        "norm_window": [15, 30, 60],
    }

    @property
    def name_with_params(self) -> str:
        sp = self.selected_params
        return f"{self.short_name}_NW{sp['norm_window']}"

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        df_tmp = df.copy()
        sp = self.selected_params
        nw = sp["norm_window"]

        # Mid price
        mid_price = (
            df_tmp[DFKey.SELL_ORDER_1_QTY_CLOSE] + df_tmp[DFKey.BUY_ORDER_1_QTY_CLOSE]
        ) / 2

        # Gradient imbalance
        obgi_raw = 0
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
        buy_order_price_dfkeys = [
            DFKey.BUY_ORDER_1_PRICE_CLOSE,
            DFKey.BUY_ORDER_2_PRICE_CLOSE,
            DFKey.BUY_ORDER_3_PRICE_CLOSE,
            DFKey.BUY_ORDER_4_PRICE_CLOSE,
            DFKey.BUY_ORDER_5_PRICE_CLOSE,
        ]
        sell_order_price_dfkeys = [
            DFKey.SELL_ORDER_1_PRICE_CLOSE,
            DFKey.SELL_ORDER_2_PRICE_CLOSE,
            DFKey.SELL_ORDER_3_PRICE_CLOSE,
            DFKey.SELL_ORDER_4_PRICE_CLOSE,
            DFKey.SELL_ORDER_5_PRICE_CLOSE,
        ]

        obgi_raw = 0
        for i in range(5):
            bid_qty = df_tmp[buy_order_dfkeys[i]]
            ask_qty = df_tmp[sell_order_dfkeys[i]]
            bid_price = df_tmp[buy_order_price_dfkeys[i]]
            ask_price = df_tmp[sell_order_price_dfkeys[i]]

            bid_term = bid_qty / ((mid_price - bid_price).abs() + 1e-8)
            ask_term = ask_qty / ((ask_price - mid_price).abs() + 1e-8)
            obgi_raw += bid_term - ask_term

        # Normalize if requested
        if nw > 1:
            obgi_smooth = (obgi_raw - obgi_raw.rolling(nw, min_periods=1).mean()) / (
                obgi_raw.rolling(nw, min_periods=1).std() + 1e-8
            )
        else:
            obgi_smooth = obgi_raw

        # Add to DataFrame
        df, col = self._add_to_df(df, obgi_smooth)
        return df, col
