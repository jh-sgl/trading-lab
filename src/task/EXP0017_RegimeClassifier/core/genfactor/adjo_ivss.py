import numpy as np
import pandas as pd

from ...util.const import DFKey, Num
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("adjo_ivss")
class AdjoIVSS(GenFactor):
    short_name = "AdjoIVSS"
    full_name = "Adjacent Option-based Implied Volatility Slope Skew"
    description = """
        Measures the asymmetry between implied volatility slopes on the call and put sides 
        using adjacent option price levels relative to FUTURE_PRICE.

        Parameters:
        - level_range: Number of adjacent levels to use in slope estimation (1~5).
        - ma_window: Optional smoothing window for reducing intra-day noise.
    """

    params = {
        "level_range": [3, 5],  # 3 = use levels 1~3; 5 = 1~5
        "ma_window": [1, 5, 10, 30, 50],  # 1 = no smoothing
    }

    @property
    def name_with_params(self) -> str:
        return f"{self.short_name}_L{self.selected_params['level_range']}_MA{self.selected_params['ma_window']}"

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        df_tmp = df.copy()
        level_range = self.selected_params["level_range"]
        ma_window = self.selected_params["ma_window"]

        # Create X: strike distances
        x = np.arange(1, level_range + 1).reshape(1, -1)

        # Collect CALL and PUT prices across the selected range
        call_keys = [
            getattr(DFKey, f"ADJ_CALL_{i}_PRICE_CLOSE")
            for i in range(1, level_range + 1)
        ]
        put_keys = [
            getattr(DFKey, f"ADJ_PUT_{i}_PRICE_CLOSE")
            for i in range(1, level_range + 1)
        ]
        call_prices = np.stack([df_tmp[k] for k in call_keys], axis=1)
        put_prices = np.stack([df_tmp[k] for k in put_keys], axis=1)

        # Calculate slope = cov(x, y) / var(x)
        x_centered = x - x.mean()
        var_x = (x_centered**2).sum()

        call_slopes = ((x_centered * call_prices).sum(axis=1)) / (var_x + Num.EPS)
        put_slopes = ((x_centered * put_prices).sum(axis=1)) / (var_x + Num.EPS)
        skew = call_slopes - put_slopes

        if ma_window > 1:
            skew = (
                pd.Series(skew, index=df_tmp.index)
                .rolling(window=ma_window, min_periods=1)
                .mean()
            )

        df, col = self._add_to_df(df, skew)
        return df, col
