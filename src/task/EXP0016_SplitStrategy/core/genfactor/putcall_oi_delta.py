import pandas as pd

from ...util.const import DFKey
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("putcall_oi_delta")
class PutCallOIDelta(GenFactor):
    short_name = "PutCallOIDelta"
    full_name = "Put-Call Open Interest Delta"
    description = """
        Measures the difference in open interest (OI) between call and put options.
    """

    params = {
        "ma_window": [1, 5, 10, 25, 50],
    }

    @property
    def name_with_params(self) -> str:
        return f"{self.short_name}_MA{self.selected_params['ma_window']}"

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        df_tmp = df.copy()

        oi_delta = df_tmp[DFKey.PUT_OPENINT_CLOSE] - df_tmp[DFKey.CALL_OPENINT_CLOSE]

        if self.selected_params["ma_window"] > 1:
            oi_delta = oi_delta.rolling(
                window=self.selected_params["ma_window"], min_periods=1
            ).mean()

        df, col = self._add_to_df(df, oi_delta)

        return df, col
