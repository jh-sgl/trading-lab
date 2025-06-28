import pandas as pd

from ...util.const import DFKey
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("adjo_vaomi")
class AdjoVAOMI(GenFactor):
    short_name = "AdjoVAOMI"
    full_name = "Adjacent Option-based Volatility-Adjusted Option Momentum Imbalance with OB Scaling"
    description = """
        Captures directional momentum imbalance between adjacent call and put options,
        adjusted for VKOSPI200 implied volatility and scaled by futures orderbook imbalance.
    """

    params = {
        "levels": [3, 5],
        "ema_span": [1, 3, 6],
        "ob_side": ["both", "buy", "sell"],
    }

    @property
    def name_with_params(self) -> str:
        p = self.selected_params
        return f"{self.short_name}_L{p['levels']}_EMA{p['ema_span']}_OB{p['ob_side'].capitalize()}"

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        df_tmp = df.copy()
        L = self.selected_params["levels"]
        ema = self.selected_params["ema_span"]
        ob_side = self.selected_params["ob_side"]

        # Momentum calculation
        call_mom = sum(
            [
                (
                    df_tmp[getattr(DFKey, f"ADJ_CALL_{i}_PRICE_CLOSE")]
                    - df_tmp[getattr(DFKey, f"ADJ_CALL_{i}_PRICE_OPEN")]
                )
                * df_tmp[getattr(DFKey, f"ADJ_CALL_{i}_VOL")]
                for i in range(1, L + 1)
            ]
        )

        put_mom = sum(
            [
                (
                    df_tmp[getattr(DFKey, f"ADJ_PUT_{i}_PRICE_CLOSE")]
                    - df_tmp[getattr(DFKey, f"ADJ_PUT_{i}_PRICE_OPEN")]
                )
                * df_tmp[getattr(DFKey, f"ADJ_PUT_{i}_VOL")]
                for i in range(1, L + 1)
            ]
        )

        raw_momentum = call_mom - put_mom

        # Volatility adjustment
        vkospi = df_tmp[DFKey.VKOSPI200_REAL_CLOSE]
        vol_adj = raw_momentum / (1 + vkospi)

        # Orderbook imbalance
        eps = 1e-6
        buy_qty = sum(
            [df_tmp[getattr(DFKey, f"BUY_ORDER_{i}_QTY_CLOSE")] for i in range(1, 6)]
        )
        sell_qty = sum(
            [df_tmp[getattr(DFKey, f"SELL_ORDER_{i}_QTY_CLOSE")] for i in range(1, 6)]
        )

        if ob_side == "both":
            ob_imb = (buy_qty - sell_qty) / (buy_qty + sell_qty + eps)
        elif ob_side == "buy":
            ob_imb = buy_qty / (buy_qty + sell_qty + eps)
        elif ob_side == "sell":
            ob_imb = -sell_qty / (buy_qty + sell_qty + eps)

        # Final score
        final_score = vol_adj * ob_imb

        # Smoothing
        if ema > 1:
            final_score = final_score.ewm(span=ema, min_periods=1).mean()

        df, col = self._add_to_df(df, final_score)

        return df, col
