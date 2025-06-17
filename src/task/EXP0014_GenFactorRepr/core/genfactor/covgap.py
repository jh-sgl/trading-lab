import numpy as np
import pandas as pd

from ...util.const import DFKey
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("covgap")
class COVGAP(GenFactor):
    short_name = "COVGAP"
    full_name = "Cross-Asset Volatility Gap Arbitrage Potential"
    description = """
        COVGAP quantifies the divergence between implied volatility signals from the KOSPI200 options
        and realized/inferred volatility expectations embedded in cross-asset instruments
        (e.g., bond futures, USD futures, VKOSPI200).
        It aims to identify arbitrage tension or regime change precursors
        that may not be visible through directional flows or liquidity imbalances alone.
    """

    params = {
        # --- Which part of IV structure to use ---
        "iv_source": [
            "avg_iv",  # (call_iv + put_iv) / 2
            "call_iv",
            "put_iv",
            "iv_skew",  # call_iv - put_iv
            "iv_slope",  # call_iv_2nd_up - call_iv_2nd_down
        ],
        # --- How to normalize IV vs VKOSPI200_real ---
        "iv_normalization_type": [
            "none",
            "zscore",
            "rolling_diff",
        ],
        "iv_normalization_window": [5, 10, 20],
        # --- Cross-asset macro divergence component ---
        "macro_component": [
            "none",
            "bond_spread",  # 167_price - 165_price
            "usd_delta",  # Δ usd_price
            "bond_spread+usd_delta",  # both
        ],
        # --- Time window for macro component change ---
        "macro_window": [
            5,  # Δ over 1 candle
            10,  # Δ over 5 candles
            20,  # Δ over 20 candles
        ],
        # --- Weighting scheme (β₁, β₂ equivalents) ---
        "weight_scheme": [
            "equal",  # 1.0 for all
            "volatility_scaled",  # scale by current VKOSPI200_real
            "zscore_scaled",  # each component z-scored
            "custom_regime",  # hand-defined rule-based weights
        ],
    }

    @property
    def name_with_params(self) -> str:
        return f"{self.short_name}_IV{self.selected_params['iv_source']}_IVN{self.selected_params['iv_normalization_type']}_IVNW{self.selected_params['iv_normalization_window']}_MC{self.selected_params['macro_component']}_WS{self.selected_params['weight_scheme']}_MW{self.selected_params['macro_window']}"

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        df_tmp = df.copy()

        # === Volatility signal (IV) ===
        iv_source = self.selected_params["iv_source"]
        if iv_source == "avg_iv":
            df_tmp["iv_signal"] = (
                df_tmp[DFKey.CALL_IV_CLOSE] + df_tmp[DFKey.PUT_IV_CLOSE]
            ) / 2
        elif iv_source == "iv_skew":
            df_tmp["iv_signal"] = (
                df_tmp[DFKey.CALL_IV_CLOSE] - df_tmp[DFKey.PUT_IV_CLOSE]
            )
        elif iv_source == "iv_slope":
            df_tmp["iv_signal"] = (
                df_tmp[DFKey.CALL_IV_2ND_UP_CLOSE] - df_tmp[DFKey.PUT_IV_2ND_DOWN_CLOSE]
            )
        elif iv_source == "call_iv":
            df_tmp["iv_signal"] = df_tmp[DFKey.CALL_IV_CLOSE]
        elif iv_source == "put_iv":
            df_tmp["iv_signal"] = df_tmp[DFKey.PUT_IV_CLOSE]
        else:
            raise ValueError(f"Invalid iv_source: {iv_source}")

        # === Normalization ===
        norm_type = self.selected_params["iv_normalization_type"]
        if norm_type == "zscore":
            df_tmp["iv_signal"] = (
                df_tmp["iv_signal"]
                - df_tmp["iv_signal"]
                .rolling(self.selected_params["iv_normalization_window"], min_periods=1)
                .mean()
            ) / (
                df_tmp["iv_signal"]
                .rolling(self.selected_params["iv_normalization_window"], min_periods=1)
                .std()
                + 1e-8
            )
        elif norm_type == "rolling_diff":
            df_tmp["iv_signal"] = df_tmp["iv_signal"] - df_tmp["iv_signal"].shift(
                self.selected_params["iv_normalization_window"]
            )
        elif norm_type == "none":
            pass
        else:
            raise ValueError(f"Invalid iv_normalization_type: {norm_type}")

        # === Macro Divergence Signal ===
        macro_signal = pd.Series(0, index=df_tmp.index)
        macro_component = self.selected_params["macro_component"]
        macro_window = self.selected_params["macro_window"]

        if "bond_spread" in macro_component:
            df_tmp["bond_spread"] = (
                df_tmp[DFKey.BOND_167_CLOSE] - df_tmp[DFKey.BOND_165_CLOSE]
            )
            bond_delta = df_tmp["bond_spread"].diff(macro_window)
            macro_signal += bond_delta

        if "usd_delta" in macro_component:
            usd_delta = df_tmp[DFKey.USD_PRICE_CLOSE].diff(macro_window)
            macro_signal += usd_delta

        # === Weighting Scheme ===
        weight_scheme = self.selected_params["weight_scheme"]
        if weight_scheme == "equal":
            beta = 1.0
        elif weight_scheme == "volatility_scaled":
            beta = df_tmp[DFKey.VKOSPI200_REAL_CLOSE] / (
                df_tmp[DFKey.VKOSPI200_REAL_CLOSE]
                .rolling(macro_window, min_periods=1)
                .mean()
                + 1e-8
            )
        elif weight_scheme == "zscore_scaled":
            beta = (
                macro_signal - macro_signal.rolling(macro_window, min_periods=1).mean()
            ) / (macro_signal.rolling(macro_window, min_periods=1).std() + 1e-8)
        elif weight_scheme == "custom_regime":
            beta = np.where(df_tmp[DFKey.VKOSPI200_REAL_CLOSE].shift(1) > 30, 1.5, 0.7)
        else:
            raise ValueError(f"Invalid weight_scheme: {weight_scheme}")

        # === Final COVGAP Calculation ===
        df_tmp["covgap_raw"] = df_tmp["iv_signal"] - df_tmp[DFKey.VKOSPI200_REAL_CLOSE]
        df_tmp["covgap"] = df_tmp["covgap_raw"] - beta * macro_signal

        # === Finalize and return ===
        df, col = self._add_to_df(df, df_tmp["covgap"])
        return df, col
