import pandas as pd

from ...util.const import DFKey, Num
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("obpd")
class OBPD(GenFactor):
    short_name = "OBPD"
    full_name = "Intrabar Orderbook-Pinned Drive"
    description = """
        Captures whether intrabar price closes near bar extreme
        in alignment or contradiction with dominant multi-level
        orderbook imbalance. Offers control over level range and
        slope-weighting scheme to emphasize shallow or deep liquidity.
    """

    params = {
        "levels": [1, 3, 5],
        "ema_span": [1, 3, 6, 12],
        "slope_weighting": ["flat", "linear", "inverse"],
    }

    @property
    def name_with_params(self) -> str:
        p = self.selected_params
        return f"{self.short_name}_L{p['levels']}_EMA{p['ema_span']}_SW{p['slope_weighting'].capitalize()}"

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        df_tmp = df.copy()
        eps = Num.EPS
        L = self.selected_params["levels"]
        ema = self.selected_params["ema_span"]
        slope = self.selected_params["slope_weighting"]

        # Bar position
        o = df_tmp[DFKey.FUTURE_PRICE_OPEN]
        h = df_tmp[DFKey.FUTURE_PRICE_HIGH]
        l = df_tmp[DFKey.FUTURE_PRICE_LOW]
        c = df_tmp[DFKey.FUTURE_PRICE_CLOSE]
        bar_pos = (c - l) / (h - l + eps)

        # Generate weights
        if slope == "flat":
            w = [1.0] * L
        elif slope == "linear":
            w = [(L - i + 1) / L for i in range(1, L + 1)]  # closer = heavier
        elif slope == "inverse":
            w = [i / L for i in range(1, L + 1)]  # deeper = heavier

        # Orderbook imbalance
        buy_weighted = sum(
            w[i] * df_tmp[getattr(DFKey, f"BUY_ORDER_{i+1}_QTY_CLOSE")]
            for i in range(L)
        )
        sell_weighted = sum(
            w[i] * df_tmp[getattr(DFKey, f"SELL_ORDER_{i+1}_QTY_CLOSE")]
            for i in range(L)
        )
        ob_imb = (buy_weighted - sell_weighted) / (buy_weighted + sell_weighted + eps)

        # Drive computation
        drive = (bar_pos - 0.5) * ob_imb

        # Smoothing
        if ema > 1:
            drive = drive.ewm(span=ema, min_periods=1).mean()

        df, col = self._add_to_df(df, drive)
        return df, col
