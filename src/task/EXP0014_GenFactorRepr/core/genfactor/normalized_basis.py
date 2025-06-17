import numpy as np
import pandas as pd

from ...util.const import DFKey
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("normalized_basis")
class NormalizedBasis(GenFactor):
    short_name = "normalized_basis"
    full_name = "Normalized Futures Basis"
    description = """
        The NormalizedBasis factor measures the relative basis (futures - spot) as a proportion 
        of the futures price, acting as a proxy for carry or storage cost signals.

        - High values: Large basis relative to futures price.
        - Low/negative values: Compressed or negative basis.
    """

    params = {"ma_window": [1, 5, 10, 20, 40, 60]}

    @property
    def name_with_params(self) -> str:
        return self.short_name

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        df_tmp = df.copy()
        normalized_basis = df_tmp[DFKey.FUTURE_BASIS_CLOSE] / (
            df_tmp[DFKey.FUTURE_PRICE_CLOSE] + 1e-8
        )

        if self.selected_params["ma_window"] > 1:
            normalized_basis = normalized_basis.rolling(
                self.selected_params["ma_window"], min_periods=1
            ).mean()

        df, col = self._add_to_df(df, normalized_basis)
        return df, col
