import pandas as pd

from ...util.const import DFKey
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("rcpskew")
class RCPSkew(GenFactor):
    short_name = "RCPSkew"
    full_name = "Relative Call-Put Transaction Skew"
    description = """
        Captures the net directional sentiment embedded in options trading by comparing
        call vs put transaction **amounts** (not volume) across adjacent strike levels.

        Definition:
        - Sum of adjusted call amounts (levels 1–3 above FUTURE_PRICE)
        - Sum of adjusted put amounts (levels 1–3 below FUTURE_PRICE)
        - Skew = (Call AMT - Put AMT) / (Call AMT + Put AMT + ε)

        Interpretation:
        - Positive: Relative preference for calls → bullish tilt
        - Negative: Relative preference for puts → bearish tilt

        This feature is normalized, making it robust across sessions and suitable for non-linear models.

        Parameters:
        - ma_window: Optional moving average smoothing window to reduce noise.
    """

    params = {
        "ma_window": [1, 5, 10, 30, 60],
    }

    @property
    def name_with_params(self) -> str:
        return f"{self.short_name}_MA{self.selected_params['ma_window']}"

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        df_tmp = df.copy()
        eps = 1e-8

        call_amt = (
            df_tmp[DFKey.ADJ_CALL_1_AMT]
            + df_tmp[DFKey.ADJ_CALL_2_AMT]
            + df_tmp[DFKey.ADJ_CALL_3_AMT]
        )
        put_amt = (
            df_tmp[DFKey.ADJ_PUT_1_AMT]
            + df_tmp[DFKey.ADJ_PUT_2_AMT]
            + df_tmp[DFKey.ADJ_PUT_3_AMT]
        )

        rcp_skew = (call_amt - put_amt) / (call_amt + put_amt + eps)

        if self.selected_params["ma_window"] > 1:
            rcp_skew = rcp_skew.rolling(
                window=self.selected_params["ma_window"], min_periods=1
            ).mean()

        df, col = self._add_to_df(df, rcp_skew)

        return df, col
