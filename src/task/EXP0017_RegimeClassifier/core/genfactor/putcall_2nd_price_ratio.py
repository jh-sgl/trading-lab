import pandas as pd

from ...util.const import DFKey, Num
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("putcall_2nd_price_ratio")
class PutCall2ndPriceRatio(GenFactor):
    short_name = "PutCall2ndPriceRatio"
    full_name = "Put-Call 2nd Price Ratio"
    description = """
        Measures the ratio of 2nd call and 2nd put options.
    """

    params = {
        "ma_window": [1, 5, 10, 25, 50],
    }

    @property
    def name_with_params(self) -> str:
        return f"{self.short_name}_MA{self.selected_params['ma_window']}"

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        df_tmp = df.copy()

        price_ratio = df_tmp[DFKey.PUT_PRICE_2ND_DOWN_CLOSE] / (
            df_tmp[DFKey.CALL_PRICE_2ND_UP_CLOSE] + Num.EPS
        )

        if self.selected_params["ma_window"] > 1:
            price_ratio = price_ratio.rolling(
                window=self.selected_params["ma_window"], min_periods=1
            ).mean()

        df, col = self._add_to_df(df, price_ratio)

        return df, col
