import pandas as pd

from ...util.const import DFKey, Num
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("adjo_gpb")
class AdjoGPB(GenFactor):
    short_name = "AdjoGPB"
    full_name = "Adjacent Option-based Intrabar Gamma Pressure"
    description = """
        Captures the interplay between intrabar futures price expansion and
        near-the-money option price structure, acting as a proxy for potential
        gamma-driven price amplification or dampening.
    """

    params = {"levels": [3, 5], "ema_span": [1, 3, 6]}

    @property
    def name_with_params(self) -> str:
        p = self.selected_params
        return f"{self.short_name}_L{p['levels']}_EMA{p['ema_span']}"

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        df_tmp = df.copy()
        L = self.selected_params["levels"]
        ema = self.selected_params["ema_span"]
        eps = Num.EPS

        # Intrabar expansion
        o = df_tmp[DFKey.FUTURE_PRICE_OPEN]
        h = df_tmp[DFKey.FUTURE_PRICE_HIGH]
        l = df_tmp[DFKey.FUTURE_PRICE_LOW]
        c = df_tmp[DFKey.FUTURE_PRICE_CLOSE]

        expansion_ratio = (c - o) / (h - l + eps)

        # Gamma bias proxy
        call_prices = sum(
            [
                df_tmp[getattr(DFKey, f"ADJ_CALL_{i}_PRICE_CLOSE")]
                for i in range(1, L + 1)
            ]
        )
        put_prices = sum(
            [
                df_tmp[getattr(DFKey, f"ADJ_PUT_{i}_PRICE_CLOSE")]
                for i in range(1, L + 1)
            ]
        )
        gamma_proxy = call_prices - put_prices

        signal = expansion_ratio * gamma_proxy

        if ema > 1:
            signal = signal.ewm(span=ema, min_periods=1).mean()

        df, col = self._add_to_df(df, signal)
        return df, col
