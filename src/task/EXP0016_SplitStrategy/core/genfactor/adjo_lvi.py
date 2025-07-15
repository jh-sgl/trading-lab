import pandas as pd

from ...util.const import DFKey, Num
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("adjo_lvi")
class AdjoLVI(GenFactor):
    short_name = "AdjoLVI"
    full_name = "Adjacent Option-based Liquidity Vacuum Index"
    description = """
        Detects liquidity vacuums by combining orderbook thinness with adjacent option
        dislocation. Useful for breakout prediction and microstructure-aware risk control.
    """

    params = {
        "levels": [3, 5],
        "normalize_depth": [True, False],
        "ema_span": [1, 3, 6],
    }

    @property
    def name_with_params(self) -> str:
        p = self.selected_params
        norm = "Norm" if p["normalize_depth"] else "Raw"
        return f"{self.short_name}_L{p['levels']}_EMA{p['ema_span']}_{norm}"

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        df_tmp = df.copy()
        L = self.selected_params["levels"]
        ema = self.selected_params["ema_span"]
        normalize = self.selected_params["normalize_depth"]
        eps = Num.EPS

        # Orderbook thinness
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
        total_depth = buy_depth + sell_depth
        thinness = 1 / (total_depth + eps)

        if normalize:
            thinness = thinness / thinness.ewm(span=20, min_periods=1).mean()

        # Option dislocation
        call_pulse = pd.concat(
            [
                df_tmp[getattr(DFKey, f"ADJ_CALL_{i}_PRICE_CLOSE")]
                - df_tmp[getattr(DFKey, f"ADJ_CALL_{i}_PRICE_OPEN")]
                for i in range(1, L + 1)
            ],
            axis=1,
        ).max(axis=1)

        put_pulse = pd.concat(
            [
                df_tmp[getattr(DFKey, f"ADJ_PUT_{i}_PRICE_CLOSE")]
                - df_tmp[getattr(DFKey, f"ADJ_PUT_{i}_PRICE_OPEN")]
                for i in range(1, L + 1)
            ],
            axis=1,
        ).max(axis=1)

        pulse = pd.concat([call_pulse, put_pulse], axis=1).max(axis=1)
        dislocation = pulse / (df_tmp[DFKey.VKOSPI200_REAL_CLOSE] + eps)

        # Final signal
        liq_vacuum = thinness * dislocation

        if ema > 1:
            liq_vacuum = liq_vacuum.ewm(span=ema, min_periods=1).mean()

        df, col = self._add_to_df(df, liq_vacuum)
        return df, col
