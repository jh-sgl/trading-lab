import numpy as np
import pandas as pd

from ...util.const import DFKey
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("adjo_sdpr")
class AdjoSDPR(GenFactor):
    short_name = "SDPR"
    full_name = "Skewed Demand Pressure Ratio"
    description = """
        Measures the imbalance between demand for OTM puts and calls, adjusted by
        the local slope of the futures order book, capturing how aggressively traders are leaning
        into one side and whether the market is absorbing it or resisting.

        Logic:
        - Compute total traded volume of OTM PUTs and CALLs (volume-based demand).
        - Compute option price slopes to proxy IV skew slope on each side.
        - Compute futures order book imbalance from depth levels (1~3).

        Final formula:
        SDPR = (PUT_vol - CALL_vol) * (PUT_slope - CALL_slope) / (BUY_OB_QTY - SELL_OB_QTY + ε)

        Interpretation:
        - Large positive: Put-heavy demand, steep downside pricing, weak buying support.
        - Large negative: Call-heavy interest, upside skew, no resistance → potential chase.
        - Near zero: Balanced demand or offsetting order book pressure.

        Parameters:
        - level_range: Strike levels used for volume and slope (1~5).
        - ob_level: Order book depth used (1~3).
        - ma_window: Smoothing window (to reduce noise).
    """

    params = {
        "level_range": [3, 5],
        "ob_level": [1, 2, 3],
        "ma_window": [1, 5, 10, 30, 60],
    }

    @property
    def name_with_params(self) -> str:
        return f"{self.short_name}_L{self.selected_params['level_range']}_OB{self.selected_params['ob_level']}_MA{self.selected_params['ma_window']}"

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        df_tmp = df.copy()
        L = self.selected_params["level_range"]
        OB = self.selected_params["ob_level"]
        MA = self.selected_params["ma_window"]

        # === Step 1: Volume Tilt ===
        call_vol_keys = [
            getattr(DFKey, f"ADJ_CALL_{i}_VOLUME") for i in range(1, L + 1)
        ]
        put_vol_keys = [getattr(DFKey, f"ADJ_PUT_{i}_VOLUME") for i in range(1, L + 1)]
        call_vol = df_tmp[call_vol_keys].sum(axis=1)
        put_vol = df_tmp[put_vol_keys].sum(axis=1)
        vol_diff = put_vol - call_vol  # net demand for puts

        # === Step 2: Price Slope Skew ===
        x = np.arange(1, L + 1).reshape(1, -1)
        x_centered = x - x.mean()
        var_x = (x_centered**2).sum()

        call_price_keys = [
            getattr(DFKey, f"ADJ_CALL_{i}_PRICE_CLOSE") for i in range(1, L + 1)
        ]
        put_price_keys = [
            getattr(DFKey, f"ADJ_PUT_{i}_PRICE_CLOSE") for i in range(1, L + 1)
        ]
        call_prices = np.stack([df_tmp[k] for k in call_price_keys], axis=1)
        put_prices = np.stack([df_tmp[k] for k in put_price_keys], axis=1)

        call_slope = ((x_centered * call_prices).sum(axis=1)) / var_x
        put_slope = ((x_centered * put_prices).sum(axis=1)) / var_x
        slope_diff = put_slope - call_slope

        # === Step 3: Order Book Pressure Imbalance ===
        buy_qty = sum(
            df_tmp[getattr(DFKey, f"BUY_ORDER_{i}_QTY_CLOSE")] for i in range(1, OB + 1)
        )
        sell_qty = sum(
            df_tmp[getattr(DFKey, f"SELL_ORDER_{i}_QTY_CLOSE")]
            for i in range(1, OB + 1)
        )
        ob_imbalance = buy_qty - sell_qty  # net buyer strength

        # === Final SDPR calculation ===
        sdpr = (vol_diff * slope_diff) / (ob_imbalance + 1e-8)

        if MA > 1:
            sdpr = (
                pd.Series(sdpr, index=df_tmp.index)
                .rolling(window=MA, min_periods=1)
                .mean()
            )

        df, col = self._add_to_df(df, sdpr)
        return df, col
