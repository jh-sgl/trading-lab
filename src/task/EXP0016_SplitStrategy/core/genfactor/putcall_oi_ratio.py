import pandas as pd

from ...util.const import DFKey, Num
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("putcall_oi_ratio")
class PutCallOIRatio(GenFactor):
    short_name = "PutCallOIRatio"
    full_name = "Put-Call Open Interest Ratio"
    description = """
        Measures the ratio of open interest (OI) between call and put options.
    """

    params = {
        "ma_window": [1, 5, 10, 25, 50],
    }

    @property
    def name_with_params(self) -> str:
        return f"{self.short_name}_MA{self.selected_params['ma_window']}"

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        df_tmp = df.copy()

        oi_ratio = df_tmp[DFKey.PUT_OPENINT_CLOSE] / (
            df_tmp[DFKey.CALL_OPENINT_CLOSE] + Num.EPS
        )

        if self.selected_params["ma_window"] > 1:
            oi_ratio = oi_ratio.rolling(
                window=self.selected_params["ma_window"], min_periods=1
            ).mean()

        df, col = self._add_to_df(df, oi_ratio)

        return df, col
