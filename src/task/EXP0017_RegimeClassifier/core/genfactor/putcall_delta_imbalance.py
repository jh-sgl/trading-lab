import pandas as pd

from ...util.const import DFKey
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("putcall_delta_imbalance")
class PutCallDeltaImbalance(GenFactor):
    short_name = "putcall_delta_imbalance"
    full_name = "Put-Call Delta Imbalance"
    description = """
        Estimates the net directional exposure from options positioning 
        using a proxy formula: (Call IV × Call OI) - (Put IV × Put OI).
        Positive values suggest bullish dealer exposure, negative bearish.
    """

    params = {
        "delta_oi_weight": [0.1, 0.3, 0.5, 0.7, 1.0],
        "ma_window": [1, 5, 10, 20, 40, 60],
    }

    @property
    def name_with_params(self) -> str:
        return (
            f"{self.short_name}_DOIW{int(self.selected_params['delta_oi_weight'] * 10)}_"
            f"MA{self.selected_params['ma_window']}"
        )

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        df_tmp = df.copy()
        delta_imbalance = (
            df_tmp[DFKey.CALL_IV_CLOSE] * df_tmp[DFKey.CALL_OPENINT_CLOSE]
            - df_tmp[DFKey.PUT_IV_CLOSE] * df_tmp[DFKey.PUT_OPENINT_CLOSE]
        )
        delta_imbalance *= self.selected_params["delta_oi_weight"]

        if self.selected_params["ma_window"] > 1:
            delta_imbalance = delta_imbalance.rolling(
                window=self.selected_params["ma_window"], min_periods=1
            ).mean()

        df, col = self._add_to_df(df, delta_imbalance)
        return df, col
