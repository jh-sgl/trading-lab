import pandas as pd

from ...util.const import DFKey
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("gamma_wall_pressure")
class GammaWallPressure(GenFactor):
    short_name = "gamma_wall_pressure"
    full_name = "Gamma Wall Pressure"
    description = """
        Captures proximity-based pressure from open interest walls in gamma-heavy strikes. 
        Computed as: (Call OI next strike + Put OI prev strike) 
        minus (At-the-money Call + Put OI), normalized by strike spacing.
    """

    params = {
        "gamma_oi_weight": [0.3, 0.5, 0.7, 1.0],
        "ma_window": [1, 5, 10, 20, 40, 60],
    }

    @property
    def name_with_params(self) -> str:
        return (
            f"{self.short_name}_GOIW{int(self.selected_params['gamma_oi_weight'] * 10)}_"
            f"MA{self.selected_params['ma_window']}"
        )

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        df_tmp = df.copy()
        gamma_wall_pressure = (
            df_tmp[DFKey.CALL_OPENINT_2ND_UP_CLOSE]
            + df_tmp[DFKey.PUT_OPENINT_2ND_DOWN_CLOSE]
            - df_tmp[DFKey.CALL_OPENINT_CLOSE]
            - df_tmp[DFKey.PUT_OPENINT_CLOSE]
        )

        gamma_wall_pressure /= 5  # assume 5 strike pricing
        gamma_wall_pressure = gamma_wall_pressure.abs()
        gamma_wall_pressure *= self.selected_params["gamma_oi_weight"]

        if self.selected_params["ma_window"] > 1:
            gamma_wall_pressure = gamma_wall_pressure.rolling(
                window=self.selected_params["ma_window"], min_periods=1
            ).mean()

        df, col = self._add_to_df(df, gamma_wall_pressure)
        return df, col
