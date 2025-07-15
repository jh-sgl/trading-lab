import numpy as np
import pandas as pd

from ...util.const import DFKey
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("casi")
class CASI(GenFactor):
    short_name = "CASI"
    full_name = "Cross-Asset Shock Absorption Index"
    description = """
        CASI captures the directional response of KOSPI200 futures to sudden price shocks 
        in exogenous macro-correlated instruments (USD, 3Y KTB, 10Y KTB). 
        
        - Measures how K200 absorbs or amplifies external shocks
        - Positive CASI: alignment with external macro shifts (risk-on coherence)
        - Negative CASI: counter-reaction or decoupling from external drivers
        - Zero CASI: no meaningful transmission detected
    """

    params = {
        "shock_response_window": [
            5,
            10,
            20,
            40,
        ],
        "vol_window_factor": [
            1,
            2,
            3,
        ],  # Realized volatility normalization window for K200
    }

    @property
    def name_with_params(self) -> str:
        sp = self.selected_params
        return f"{self.short_name}_SRW{sp['shock_response_window']}_VW{sp['vol_window_factor']}"

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        df_tmp = df.copy()
        sp = self.selected_params
        srw = sp["shock_response_window"]
        vw = sp["vol_window_factor"] * srw

        # 1. Compute shock detection for macro assets (z-scored log returns)
        def zshock(col):
            log_ret = np.log(df_tmp[col] + 1e-8).diff()
            return (log_ret - log_ret.rolling(srw, min_periods=1).mean()) / (
                log_ret.rolling(srw, min_periods=1).std() + 1e-8
            )

        z_usd = zshock(DFKey.USD_PRICE_CLOSE)
        z_3y = zshock(DFKey.BOND_165_CLOSE)
        z_10y = zshock(DFKey.BOND_167_CLOSE)

        # 2. Aggregate direction of exogenous shock
        shock_strength = z_usd + z_3y + z_10y

        # 3. K200 response: normalized future return over response window
        k200_ret = df_tmp[DFKey.FUTURE_PRICE_CLOSE].pct_change(srw)
        realized_vol = (
            df_tmp[DFKey.FUTURE_PRICE_CLOSE]
            .pct_change()
            .rolling(vw, min_periods=1)
            .std()
        )
        response_norm = k200_ret / (realized_vol + 1e-8)

        # 4. CASI = sign(shock) × response magnitude
        casi_raw = np.sign(shock_strength.shift(srw)) * response_norm

        # 5. Add to DataFrame
        df, col = self._add_to_df(df, casi_raw)
        return df, col
