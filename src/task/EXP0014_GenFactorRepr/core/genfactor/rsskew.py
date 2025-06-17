import logging
import numpy as np
import pandas as pd

from ...util.const import DFKey
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("rsskew")
class RSSkew(GenFactor):
    short_name = "RSSkew"
    full_name = "Realized Skewness Signal"
    description = """
        Realized Skewness Signal (RSSkew) captures third-order asymmetry in high-frequency return distributions.
        This signal is orthogonal to volatility and pivot-based signals by focusing on the skewness (directional bias)
        of recent price action.

        - Uses rolling skewness over log returns.
        - Captures latent imbalance in return distribution.
        - Useful for detecting early directional commitment or hidden accumulation.
    """

    params = {
        "skew_window": [5, 10, 20, 30],  # window size for realized skewness
        "norm_window": [10, 30, 60],  # for rolling mean normalization
    }

    @property
    def name_with_params(self) -> str:
        sp = self.selected_params
        return f"{self.short_name}_SW{sp['skew_window']}_NW{sp['norm_window']}"

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        df_tmp = df.copy()
        sp = self.selected_params
        price = df_tmp[DFKey.FUTURE_PRICE_CLOSE]

        # 1. Compute log returns
        log_ret = np.log(price / price.shift(1))

        # 2. Rolling skewness (realized)
        def skew_func(x):
            mu = np.mean(x)
            std = np.std(x)
            if std < 1e-8:
                return 0.0
            return np.mean((x - mu) ** 3) / (std**3)

        skewness = log_ret.rolling(sp["skew_window"], min_periods=1).apply(
            skew_func, raw=True
        )

        # 3. Normalize if requested
        if sp["norm_window"] > 1:
            skewness_smooth = skewness.rolling(sp["norm_window"], min_periods=1).mean()
        else:
            skewness_smooth = skewness

        # 4. Output
        df, col = self._add_to_df(df, skewness_smooth)
        return df, col
