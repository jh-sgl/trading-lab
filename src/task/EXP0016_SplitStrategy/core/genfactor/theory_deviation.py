import numpy as np
import pandas as pd

from ...util.const import DFKey
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("theory_deviation")
class TheoryDeviation(GenFactor):
    short_name = "theory_deviation"
    full_name = "Theoretical Deviation"
    description = """
        The TheoryDeviation factor captures the percentage deviation of actual futures prices 
        from their theoretical valuation. It provides insight into market inefficiency or pricing biases.

        - Positive values: Futures are trading above theoretical value (potential overpricing).
        - Negative values: Futures are trading below theoretical value (potential underpricing).
        - Zero: Futures price aligns with theory.
    """

    params = {"ma_window": [1, 5, 10, 20, 40, 60]}

    @property
    def name_with_params(self) -> str:
        return f"{self.short_name}_MA{self.selected_params['ma_window']}"

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        df_tmp = df.copy()
        theory_deviation = (
            df_tmp[DFKey.FUTURE_PRICE_CLOSE] - df_tmp[DFKey.FUTURE_THEORY_CLOSE]
        ) / df_tmp[DFKey.FUTURE_PRICE_CLOSE]

        if self.selected_params["ma_window"] > 1:
            theory_deviation = theory_deviation.rolling(
                self.selected_params["ma_window"], min_periods=1
            ).mean()

        df, col = self._add_to_df(df, theory_deviation)
        return df, col
