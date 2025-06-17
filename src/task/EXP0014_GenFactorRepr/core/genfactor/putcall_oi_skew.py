import pandas as pd

from ...util.const import DFKey
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("putcall_oi_skew")
class PutCallOISkew(GenFactor):
    short_name = "PutCallOISkew"
    full_name = "Put-Call Open Interest Skew"
    description = """
        Measures skew in open interest (OI) between call and put options across strike distances,
        serving as a proxy for directional positioning or dealer inventory pressure.

        The skew is calculated as:
        - Difference in OI at higher strikes (calls) versus OI at lower strikes (puts).
        - Optionally smoothed using a moving average to reduce noise.

        Interpretation:
        - Positive skew: More aggressive call buying / unwinding put protection.
        - Negative skew: Build-up in put-side hedging or bearish speculation.

        Parameters:
        - ma_window: Smoothing window (in days) to average the skew values (default 1 = no smoothing).
    """

    params = {
        "ma_window": [1, 5, 10, 25, 50],
    }

    @property
    def name_with_params(self) -> str:
        return f"{self.short_name}_MA{self.selected_params['ma_window']}"

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        df_tmp = df.copy()

        call_oi_skew = (
            df_tmp[DFKey.CALL_OPENINT_2ND_UP_CLOSE] - df_tmp[DFKey.CALL_OPENINT_CLOSE]
        )
        put_oi_skew = (
            df_tmp[DFKey.PUT_OPENINT_CLOSE] - df_tmp[DFKey.PUT_OPENINT_2ND_DOWN_CLOSE]
        )
        oi_skew = call_oi_skew - put_oi_skew

        if self.selected_params["ma_window"] > 1:
            oi_skew = oi_skew.rolling(
                window=self.selected_params["ma_window"], min_periods=1
            ).mean()

        df, col = self._add_to_df(df, oi_skew)

        return df, col
