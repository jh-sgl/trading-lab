import pandas as pd

from ...util.const import DFKey
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("imri")
class IMRI(GenFactor):
    short_name = "IMRI"
    full_name = "Instantaneous Market Response Index"
    description = """
        IMRI captures bar-level reflexivity of the KOSPI200 futures market by measuring
        normalized price and liquidity responses to strong directional order flow imbalances.

        - All components are normalized with robust z-scores for scale alignment.
        - Shock: Based on sharp change in net trade imbalance.
        - Response: Includes price impact, order book liquidity withdrawal, and reversion tendencies.
        - High IMRI: Aggressive flow drives price with poor liquidity support.
    """

    params = {
        "shock_quantile": [0.8, 0.9, 0.95],
        "shock_lookback": [5, 10, 20, 30],
        "normalization_window": [10, 30, 60],
        "price_weight": [0.5, 0.6, 0.7],
        "liquidity_weight": [0.2, 0.3],
        "reversion_weight": [0.1, 0.2, 0.3],
        "imri_window": [1, 3, 5],
    }

    @property
    def name_with_params(self) -> str:
        sp = self.selected_params
        return (
            f"{self.short_name}_Q{int(sp['shock_quantile'] * 100)}_"
            f"LB{sp['shock_lookback']}_NW{sp['normalization_window']}_"
            f"PW{int(sp['price_weight']*10)}_"
            f"LW{int(sp['liquidity_weight']*10)}_"
            f"RW{int(sp['reversion_weight']*10)}_"
            f"IMRI{sp['imri_window']}"
        )

    @staticmethod
    def robust_zscore(series: pd.Series, window: int) -> pd.Series:
        median = series.rolling(window, min_periods=1).median()
        mad = (series - median).abs().rolling(window, min_periods=1).median() + 1e-8
        return (series - median) / mad

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        df_tmp = df.copy()
        sp = self.selected_params

        # 1. Net flow and shock strength
        df_tmp["net_flow"] = df_tmp[DFKey.TRADE_CUMSUM_CLOSE]
        df_tmp["shock_strength"] = df_tmp["net_flow"].diff()

        # 2. Shock filter based on quantile
        threshold = (
            df_tmp["shock_strength"]
            .abs()
            .rolling(sp["shock_lookback"], min_periods=1)
            .quantile(sp["shock_quantile"])
        )
        df_tmp["is_shock"] = df_tmp["shock_strength"].abs() > threshold

        # 3. Components
        df_tmp["price_response"] = df_tmp[DFKey.FUTURE_PRICE_CLOSE].pct_change()
        df_tmp["depth_total"] = (
            df_tmp[DFKey.BUY_ORDER_1_QTY_CLOSE]
            + df_tmp[DFKey.BUY_ORDER_2_QTY_CLOSE]
            + df_tmp[DFKey.BUY_ORDER_3_QTY_CLOSE]
            + df_tmp[DFKey.BUY_ORDER_4_QTY_CLOSE]
            + df_tmp[DFKey.BUY_ORDER_5_QTY_CLOSE]
        ) + (
            df_tmp[DFKey.SELL_ORDER_1_QTY_CLOSE]
            + df_tmp[DFKey.SELL_ORDER_2_QTY_CLOSE]
            + df_tmp[DFKey.SELL_ORDER_3_QTY_CLOSE]
            + df_tmp[DFKey.SELL_ORDER_4_QTY_CLOSE]
            + df_tmp[DFKey.SELL_ORDER_5_QTY_CLOSE]
        )
        df_tmp["liq_withdraw"] = -df_tmp["depth_total"].diff()
        df_tmp["reversion_score"] = -df_tmp["price_response"] * df_tmp[
            "shock_strength"
        ].shift(1)

        # 4. Normalize each component
        norm_price = self.robust_zscore(
            df_tmp["price_response"], sp["normalization_window"]
        )
        norm_liq = self.robust_zscore(
            df_tmp["liq_withdraw"], sp["normalization_window"]
        )
        norm_rev = self.robust_zscore(
            df_tmp["reversion_score"], sp["normalization_window"]
        )

        # 5. Compute weighted sum
        df_tmp["imri_raw"] = (
            sp["price_weight"] * norm_price
            + sp["liquidity_weight"] * norm_liq
            + sp["reversion_weight"] * norm_rev
        )

        # 6. Filter only shock events
        df_tmp["imri_filtered"] = df_tmp["imri_raw"].where(df_tmp["is_shock"]).fillna(0)

        # 7. Smooth the index
        if sp["imri_window"] > 1:
            df_tmp["imri_smooth"] = (
                df_tmp["imri_filtered"].rolling(sp["imri_window"], min_periods=1).mean()
            )
        else:
            df_tmp["imri_smooth"] = df_tmp["imri_filtered"]

        df, col = self._add_to_df(df, df_tmp["imri_smooth"])
        return df, col
