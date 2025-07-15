import pandas as pd

from ...util.const import DFKey, Num
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("iblsi")
class IBLSI(GenFactor):
    short_name = "IBLSI"
    full_name = "Intrabar Liquidity Sweep and Price Imbalance"
    description = """
        Captures whether intra-bar price movement is aligned with or
        in defiance of orderbook liquidity, suggesting aggressive flow or absorption.
    """

    params = {"levels": [3, 5], "ema_span": [1, 3, 6, 12]}

    @property
    def name_with_params(self) -> str:
        p = self.selected_params
        return f"{self.short_name}_L{p['levels']}_EMA{p['ema_span']}"

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        df_tmp = df.copy()
        L = self.selected_params["levels"]
        ema = self.selected_params["ema_span"]
        eps = Num.EPS

        # Price movement inside bar
        o = df_tmp[DFKey.FUTURE_PRICE_OPEN]
        h = df_tmp[DFKey.FUTURE_PRICE_HIGH]
        l = df_tmp[DFKey.FUTURE_PRICE_LOW]
        c = df_tmp[DFKey.FUTURE_PRICE_CLOSE]

        price_pressure = (c - o) / (h - l + eps)

        # Orderbook depth imbalance
        buy_depth = sum(
            [
                df_tmp[getattr(DFKey, f"BUY_ORDER_{i}_QTY_CLOSE")]
                for i in range(1, L + 1)
            ]
        )
        sell_depth = sum(
            [
                df_tmp[getattr(DFKey, f"SELL_ORDER_{i}_QTY_CLOSE")]
                for i in range(1, L + 1)
            ]
        )
        book_imbalance = (buy_depth - sell_depth) / (buy_depth + sell_depth + eps)

        # Liquidity sweep indicator
        signal = price_pressure * (-book_imbalance)

        if ema > 1:
            signal = signal.ewm(span=ema, min_periods=1).mean()

        df, col = self._add_to_df(df, signal)
        return df, col
