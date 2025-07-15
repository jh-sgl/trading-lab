import pandas as pd

from ...util.const import DFKey
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("book_flicker_rate")
class BookFlickerRate(GenFactor):
    short_name = "book_flicker_rate"
    full_name = "Book Flicker Rate"
    description = """
        Book Flicker Rate (BFR) quantifies the instability of book liquidity
        by capturing how frequently the bid/ask values change at each depth.
        High flicker rates suggest transient or synthetic liquidity, typical in HFT-dominated environments.
    """

    params = {"depth": [1, 2, 3, 4, 5], "flicker_window": [2, 3, 5]}

    @property
    def name_with_params(self) -> str:
        return f"{self.short_name}_FL{self.selected_params['flicker_window']}_D{self.selected_params['depth']}"

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        df_tmp = df.copy()

        buy_order_keys = [
            DFKey.BUY_ORDER_1_QTY_CLOSE,
            DFKey.BUY_ORDER_2_QTY_CLOSE,
            DFKey.BUY_ORDER_3_QTY_CLOSE,
            DFKey.BUY_ORDER_4_QTY_CLOSE,
            DFKey.BUY_ORDER_5_QTY_CLOSE,
        ]
        sell_order_keys = [
            DFKey.SELL_ORDER_1_QTY_CLOSE,
            DFKey.SELL_ORDER_2_QTY_CLOSE,
            DFKey.SELL_ORDER_3_QTY_CLOSE,
            DFKey.SELL_ORDER_4_QTY_CLOSE,
            DFKey.SELL_ORDER_5_QTY_CLOSE,
        ]

        buy = pd.Series(0, index=df_tmp.index)
        sell = pd.Series(0, index=df_tmp.index)
        for depth in range(self.selected_params["depth"]):
            buy += df_tmp[buy_order_keys[depth]]
            sell += df_tmp[sell_order_keys[depth]]

        flicker = (
            (buy.pct_change().abs() + sell.pct_change().abs())
            .rolling(window=self.selected_params["flicker_window"], min_periods=1)
            .mean()
        )
        bfr = flicker.fillna(0)

        df, col = self._add_to_df(df, bfr)
        return df, col
