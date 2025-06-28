import numpy as np
import pandas as pd

from ...util.const import DFKey
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("ird")
class IRD(GenFactor):
    short_name = "ird"
    full_name = "Imbalance Reversal Divergence"
    description = """
        Imbalance Reversal Divergence (IRD) measures the failure of order book imbalance
        to result in price movement. It highlights divergence between expected and realized
        market response, flagging environments where visible pressure is deceptive.
    """

    params = {"ird_window": [3, 5, 10, 20], "order_depth": [1, 2, 3, 4, 5]}

    @property
    def name_with_params(self) -> str:
        return f"{self.short_name}_W{self.selected_params['ird_window']}"

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
        for depth in range(self.selected_params["order_depth"]):
            buy += df_tmp[buy_order_keys[depth]]
            sell += df_tmp[sell_order_keys[depth]]

        imbalance = buy - sell
        imbalance_sign = imbalance.shift(1).fillna(0).apply(np.sign)

        price_change = df_tmp[DFKey.FUTURE_PRICE_CLOSE].diff().fillna(0)
        price_sign = price_change.apply(np.sign)

        mismatch = (imbalance_sign != price_sign).astype(int)
        ird = mismatch.rolling(self.selected_params["ird_window"]).mean().fillna(0)

        df, col = self._add_to_df(df, ird)
        return df, col
