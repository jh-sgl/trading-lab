import pandas as pd

from ...util.const import DFKey
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("vanna_pressure")
class VannaPressure(GenFactor):
    short_name = "VannaPressure"
    full_name = "Vanna-Driven Sentiment Pressure"
    description = """
        Estimates directional sentiment pressure driven by vanna exposure.

        The vanna proxy is computed as:
        - The product of implied volatility skew slope and the implied volatility gap between calls and puts.
        - This reflects how dealer re-hedging activity may intensify or dampen due to changes in skew and vol asymmetry.

        Interpretation:
        - Captures nonlinear dealer flows related to changes in delta and vega exposure.
        - Positive values may indicate pressure to buy into upward moves, negative for downward.

        Parameters:
        - normalize_iv_gap: Whether to normalize the IV gap by the total implied volatility.
        - ma_window: Optional smoothing of the vanna signal (default 1 = no smoothing).
    """

    params = {
        "ma_window": [1, 5, 10, 25, 50],
        "normalize_iv_gap": [True, False],
    }

    @property
    def name_with_params(self) -> str:
        return f"{self.short_name}_MA{self.selected_params['ma_window']}"

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        df_tmp = df.copy()

        call_skew = (
            df_tmp[DFKey.CALL_IV_2ND_UP_CLOSE] - df_tmp[DFKey.CALL_IV_CLOSE]
        ) / 5
        put_skew = (
            df_tmp[DFKey.PUT_IV_CLOSE] - df_tmp[DFKey.PUT_IV_2ND_DOWN_CLOSE]
        ) / 5
        skew_slope = call_skew - put_skew

        iv_gap = df_tmp[DFKey.CALL_IV_CLOSE] - df_tmp[DFKey.PUT_IV_CLOSE]
        if self.selected_params["normalize_iv_gap"]:
            iv_gap = iv_gap / (
                df_tmp[DFKey.CALL_IV_CLOSE] + df_tmp[DFKey.PUT_IV_CLOSE] + 1e-8
            )
        vanna_pressure = skew_slope * iv_gap

        df_tmp["vanna_pressure_raw"] = vanna_pressure

        if self.selected_params["ma_window"] > 1:
            df_tmp["vanna_pressure_raw"] = (
                df_tmp["vanna_pressure_raw"]
                .rolling(window=self.selected_params["ma_window"], min_periods=1)
                .mean()
            )

        df, col = self._add_to_df(df, df_tmp["vanna_pressure_raw"])

        return df, col
