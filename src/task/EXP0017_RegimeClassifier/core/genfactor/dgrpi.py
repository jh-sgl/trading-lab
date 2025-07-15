import pandas as pd
import numpy as np

from ...util.const import DFKey
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("dgrpi")
class DGRPI(GenFactor):
    short_name = "DGRPI"
    full_name = "Delta-Gamma Realignment Pressure Index"
    description = """
        The Delta-Gamma Realignment Pressure Index captures the hedging flows arising 
        from misalignment in delta and gamma exposures across key option strikes. 
        It reflects the pressure on underlying futures price caused by dealers’ rebalancing 
        activity as delta/gamma positioning becomes increasingly asymmetric.
        
        - Delta imbalance: Net directional exposure from options positioning.
        - Gamma wall proximity: Price proximity to high gamma-OI zones.
        - Dynamic response: Weighted by recent futures price velocity.
    """

    params = {
        "gamma_oi_weight": [0.3, 0.5, 0.7, 1.0],
        "delta_oi_weight": [0.1, 0.3, 0.5, 0.7],
        "price_return_window": [1, 3, 5],
        "strike_spacing": [5, 10],
        "ma_window": [1, 10],
    }

    @property
    def name_with_params(self) -> str:
        return (
            f"{self.short_name}_PR{self.selected_params['price_return_window']}_"
            f"GOIW{int(self.selected_params['gamma_oi_weight'] * 10)}_"
            f"DOIW{int(self.selected_params['delta_oi_weight'] * 10)}_"
            f"MA{self.selected_params['ma_window']}"
        )

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        df_tmp = df.copy()

        # Proxy delta = IV × OI (simplified directional signal)
        delta_imbalance = (
            df_tmp[DFKey.CALL_IV_CLOSE] * df_tmp[DFKey.CALL_OPENINT_CLOSE]
        ) - (df_tmp[DFKey.PUT_IV_CLOSE] * df_tmp[DFKey.PUT_OPENINT_CLOSE])

        # Proxy gamma pressure = sum of gamma-weighted OI difference across strikes
        gamma_wall_pressure = (
            df_tmp[DFKey.CALL_OPENINT_2ND_UP_CLOSE]
            + df_tmp[DFKey.PUT_OPENINT_2ND_DOWN_CLOSE]
        ) - (df_tmp[DFKey.CALL_OPENINT_CLOSE] + df_tmp[DFKey.PUT_OPENINT_CLOSE])

        # Normalize gamma wall pressure by strike spacing (2025-05-28: 5 units; can select from 1, 5, 10)
        gamma_wall_pressure = (
            gamma_wall_pressure / self.selected_params["strike_spacing"]
        )

        # Price return signal (directional context)
        raw_return = (
            df_tmp[DFKey.FUTURE_PRICE_CLOSE]
            .pct_change(self.selected_params["price_return_window"])
            .fillna(0)
        )

        directional_weight = np.sign(raw_return) * np.log1p(np.abs(raw_return))

        # Combine into DGRPI raw score
        df_tmp["dgrpi_raw"] = (
            self.selected_params["delta_oi_weight"]
            * delta_imbalance
            * directional_weight
            + self.selected_params["gamma_oi_weight"]
            * gamma_wall_pressure.abs()
            * directional_weight
        )

        if self.selected_params["ma_window"] > 1:
            df_tmp["dgrpi_raw"] = (
                df_tmp["dgrpi_raw"]
                .rolling(window=self.selected_params["ma_window"], min_periods=1)
                .mean()
            )

        df, col = self._add_to_df(df, df_tmp["dgrpi_raw"])

        return df, col
