import pandas as pd

from ...util.const import DFKey
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("pldi")
class PLDI(GenFactor):
    short_name = "PLDI"
    full_name = "Price-Liquidity Divergence Index"
    description = """
        PLDI captures divergence between short-term price changes and shifts in order book liquidity depth.

        - Positive PLDI: price rises while sell-side liquidity strengthens (bullish absorption)
        - Negative PLDI: price drops while buy-side liquidity strengthens (bearish pressure)
        - Highlights hidden strength/weakness when price moves against visible liquidity shift
    """

    params = {
        "price_window": [1, 5, 10, 20, 30],  # ΔP window
        "liquidity_window": [12, 24],  # ΔL window
    }

    @property
    def name_with_params(self) -> str:
        sp = self.selected_params
        return f"{self.short_name}_PW{sp['price_window']}_LW{sp['liquidity_window']}"

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        df_tmp = df.copy()
        sp = self.selected_params
        pw = sp["price_window"]
        lw = sp["liquidity_window"]

        # Price momentum ΔP
        delta_price = df_tmp[DFKey.FUTURE_PRICE_CLOSE].diff(pw)

        # Liquidity imbalance at each time
        buy_order_dfkeys = [
            DFKey.BUY_ORDER_1_CLOSE,
            DFKey.BUY_ORDER_2_CLOSE,
            DFKey.BUY_ORDER_3_CLOSE,
            DFKey.BUY_ORDER_4_CLOSE,
            DFKey.BUY_ORDER_5_CLOSE,
        ]
        sell_order_dfkeys = [
            DFKey.SELL_ORDER_1_CLOSE,
            DFKey.SELL_ORDER_2_CLOSE,
            DFKey.SELL_ORDER_3_CLOSE,
            DFKey.SELL_ORDER_4_CLOSE,
            DFKey.SELL_ORDER_5_CLOSE,
        ]

        buy_liquidity = df_tmp[buy_order_dfkeys].sum(axis=1)
        sell_liquidity = df_tmp[sell_order_dfkeys].sum(axis=1)
        liquidity_imbalance = (buy_liquidity - sell_liquidity) / (
            buy_liquidity + sell_liquidity + 1e-8
        )

        # Change in liquidity imbalance ΔL
        delta_liquidity = liquidity_imbalance.diff(lw)

        # PLDI = ΔP × (-ΔL)
        pldi = delta_price * (-delta_liquidity)

        # Add to DataFrame
        df, col = self._add_to_df(df, pldi)
        return df, col
