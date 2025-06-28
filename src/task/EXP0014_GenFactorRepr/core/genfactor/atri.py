import pandas as pd

from ...util.const import DFKey
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("atri")
class ATRI(GenFactor):
    short_name = "ATRI"
    full_name = "Asynchronous Trade Reaction Index"
    description = """
        ATRI captures trade-liquidity response latency by measuring the divergence between 
        aggressive net flow changes and the corresponding reaction from the order book.

        - Trade Pulse: Captures aggressive directional net flow.
        - Book Lag: Detects delayed liquidity response at best bid/ask.
        - Asymmetry: Measures if response latency differs by direction.
        - Adaptive Thresholding: Uses rolling quantile to define meaningful net flow impulses.
        - High ATRI: Suggests informed trading or short-term inefficiency via slow liquidity replenishment.
    """

    params = {
        "reaction_window": [3, 5],
        "normalization_window": [30, 60, 90],
        "flow_quantile": [0.8, 0.9, 0.95],
        "quantile_window_factor": [5, 10, 20],
    }

    @property
    def name_with_params(self) -> str:
        sp = self.selected_params
        return (
            f"{self.short_name}_RW{sp['reaction_window']}_NW{sp['normalization_window']}_"
            f"Q{int(sp['flow_quantile'] * 100)}"
        )

    @staticmethod
    def robust_zscore(series: pd.Series, window: int) -> pd.Series:
        median = series.rolling(window, min_periods=1).median()
        mad = (series - median).abs().rolling(window, min_periods=1).median() + 1e-8
        return (series - median) / mad

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        df_tmp = df.copy()
        sp = self.selected_params

        # 1. Trade Pulse: net directional flow change (causal)
        df_tmp["net_flow"] = df_tmp[DFKey.TRADE_CUMSUM_CLOSE].diff()

        # 2. Shifted net flow to create causal impulse trigger
        df_tmp["past_net_flow"] = df_tmp["net_flow"].shift(sp["reaction_window"])

        # 3. Order Book Reaction: reaction *after* impulse
        df_tmp["buy_1_diff"] = df_tmp[DFKey.BUY_ORDER_1_QTY_CLOSE] - df_tmp[
            DFKey.BUY_ORDER_1_QTY_CLOSE
        ].shift(sp["reaction_window"])
        df_tmp["sell_1_diff"] = df_tmp[DFKey.SELL_ORDER_1_QTY_CLOSE] - df_tmp[
            DFKey.SELL_ORDER_1_QTY_CLOSE
        ].shift(sp["reaction_window"])

        # 4. Adaptive threshold for pulse detection
        q_threshold = (
            df_tmp["past_net_flow"]
            .abs()
            .rolling(
                sp["reaction_window"] * sp["quantile_window_factor"], min_periods=1
            )
            .quantile(sp["flow_quantile"])
        )
        df_tmp["is_pulse"] = df_tmp["past_net_flow"].abs() > q_threshold

        # 5. Asymmetric book lag (causal, post-pulse reaction)
        df_tmp["buy_side_atri"] = df_tmp["past_net_flow"].where(
            df_tmp["past_net_flow"] > 0
        ) * (-df_tmp["sell_1_diff"])
        df_tmp["sell_side_atri"] = df_tmp["past_net_flow"].where(
            df_tmp["past_net_flow"] < 0
        ) * (-df_tmp["buy_1_diff"])

        # 6. Raw ATRI value
        df_tmp["atri_raw"] = df_tmp["buy_side_atri"].fillna(0) + df_tmp[
            "sell_side_atri"
        ].fillna(0)

        # 7. Normalization
        df_tmp["atri_norm"] = self.robust_zscore(
            df_tmp["atri_raw"], sp["normalization_window"]
        )

        # 8. Apply signal mask based on adaptive pulse filter
        df_tmp["atri_filtered"] = df_tmp["atri_norm"].where(df_tmp["is_pulse"])

        # 9. Smooth over reaction window
        df_tmp["atri_smooth"] = (
            df_tmp["atri_filtered"]
            .rolling(sp["reaction_window"], min_periods=1)
            .mean()
            .fillna(0)
        )

        # 10. Finalize
        df, col = self._add_to_df(df, df_tmp["atri_smooth"])
        return df, col
