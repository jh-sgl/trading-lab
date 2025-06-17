import numpy as np
import pandas as pd

from ...util.const import DFKey
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("LSOFI")
class LSOFI(GenFactor):
    short_name = "LSOFI"
    full_name = "Liquidity-Skewed Order Flow Imbalance"
    description = """
        This feature aims to capture the directional intent of the market,
        filtered through actual tradable liquidity,
        and conditioned on the market's perceived future volatility (from options).
        It can indicate whether price pressure is likely to result in movement
        or be absorbed—crucial for forecasting price dynamics.
    """

    params = {
        "order_book_depth": [1, 2, 3, 4, 5],
        "ma_window": [1, 5, 20, 60],
        "volatility_source": [
            "avg_iv",
            "call_iv",
            "put_iv",
            "none",
        ],
        "signed_participant_flow": [True, False],
    }

    @property
    def name_with_params(self) -> str:
        return f"{self.short_name}_OB{self.selected_params['order_book_depth']}_MA{self.selected_params['ma_window']}_VS{self.selected_params['volatility_source']}_{'SP' if self.selected_params['signed_participant_flow'] else 'NSP'}"

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        df_tmp = df.copy()

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

        buy_qty_cols = [
            buy_order_dfkeys[i]
            for i in range(0, self.selected_params["order_book_depth"])
        ]
        sell_qty_cols = [
            sell_order_dfkeys[i]
            for i in range(0, self.selected_params["order_book_depth"])
        ]

        df_tmp["buy_liquidity"] = df_tmp[buy_qty_cols].sum(axis=1)
        df_tmp["sell_liquidity"] = df_tmp[sell_qty_cols].sum(axis=1)

        # Net aggressive order flow
        df_tmp["net_flow"] = df_tmp[DFKey.TRADE_CUMSUM_CLOSE].diff()

        # Liquidity-skewed imbalance core
        df_tmp["lsofi_raw"] = (
            df_tmp["net_flow"]
            / (df_tmp["buy_liquidity"] + df_tmp["sell_liquidity"] + 1e-8)
        ) * (df_tmp["sell_liquidity"] - df_tmp["buy_liquidity"])

        # Smoothing
        if self.selected_params["ma_window"] > 1:
            df_tmp["lsofi_raw"] = (
                df_tmp["lsofi_raw"]
                .rolling(window=self.selected_params["ma_window"], min_periods=1)
                .mean()
            )

        # Volatility context
        if self.selected_params["volatility_source"] == "avg_iv":
            df_tmp["volatility"] = (
                df_tmp[DFKey.CALL_IV_CLOSE] + df_tmp[DFKey.PUT_IV_CLOSE]
            ) / 2
        elif self.selected_params["volatility_source"] == "call_iv":
            df_tmp["volatility"] = df_tmp[DFKey.CALL_IV_CLOSE]
        elif self.selected_params["volatility_source"] == "put_iv":
            df_tmp["volatility"] = df_tmp[DFKey.PUT_IV_CLOSE]
        elif self.selected_params["volatility_source"] == "none":
            df_tmp["volatility"] = 1.0
        else:
            raise ValueError(
                f"Invalid volatility source: {self.selected_params['volatility_source']}"
            )

        df_tmp["lsofi_vol"] = df_tmp["lsofi_raw"] * df_tmp["volatility"]

        # Optional participant polarity
        if self.selected_params["signed_participant_flow"]:
            df_tmp["participant_sign"] = np.sign(
                df_tmp[DFKey.FOREIGN_TRADE_CLOSE] - df_tmp[DFKey.INDIVIDUAL_TRADE_CLOSE]
            )
            df_tmp["lsofi"] = df_tmp["lsofi_vol"] * df_tmp["participant_sign"]
        else:
            df_tmp["lsofi"] = df_tmp["lsofi_vol"]

        df, col = self._add_to_df(df, df_tmp["lsofi"])

        return df, col
