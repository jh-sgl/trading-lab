import pandas as pd

from ...util.const import DFKey
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("trade_pulse")
class TradePulse(GenFactor):
    short_name = "TradePulse"
    full_name = "Aggressive Net Flow Pulse"

    params = {
        "reaction_window": [3, 5, 10, 30],
        "quantile_window_factor": [5, 10, 20],
        "flow_quantile": [0.8, 0.9, 0.95],
    }

    @property
    def name_with_params(self) -> str:
        return f"{self.short_name}_RW{self.selected_params['reaction_window']}"

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        sp = self.selected_params
        df_tmp = df.copy()

        df_tmp["net_flow"] = df_tmp[DFKey.TRADE_CUMSUM_CLOSE].diff()
        df_tmp["past_net_flow"] = df_tmp["net_flow"].shift(sp["reaction_window"])

        q_threshold = (
            df_tmp["past_net_flow"]
            .abs()
            .rolling(
                sp["reaction_window"] * sp["quantile_window_factor"], min_periods=1
            )
            .quantile(sp["flow_quantile"])
        )
        df_tmp["is_pulse"] = (df_tmp["past_net_flow"].abs() > q_threshold).astype(int)

        df, col = self._add_to_df(df, df_tmp["is_pulse"])
        return df, col
