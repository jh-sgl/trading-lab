import pandas as pd

from ...util.const import DFKey, Num
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("volcr")
class VolCR(GenFactor):
    short_name = "VolCR"
    full_name = "Volatility Concentration Ratio"
    description = """
        Measures how concentrated the options trading activity is near the current FUTURE_PRICE,
        serving as a proxy for focus or dispersion of speculative/hedging interest.

        Computation:
        - Uses transaction *volume* (not amount) across ADJ_{CALL,PUT}_{1~5}_VOLUME.
        - VolCR = Volume at level 1 / (Sum of volume at levels 1 through 5)

        Interpretation:
        - High VolCR → trading highly concentrated near ATM → short-term focus, gamma hedging
        - Low VolCR → activity dispersed across OTM strikes → longer-term bets or skew plays

        Parameters:
        - ma_window: Optional smoothing window to reduce noise.
    """

    params = {
        "ma_window": [1, 3, 5, 10, 30, 60],
    }

    @property
    def name_with_params(self) -> str:
        return f"{self.short_name}_MA{self.selected_params['ma_window']}"

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        df_tmp = df.copy()

        call_vols = [
            df_tmp[DFKey.ADJ_CALL_1_VOLUME],
            df_tmp[DFKey.ADJ_CALL_2_VOLUME],
            df_tmp[DFKey.ADJ_CALL_3_VOLUME],
            df_tmp[DFKey.ADJ_CALL_4_VOLUME],
            df_tmp[DFKey.ADJ_CALL_5_VOLUME],
        ]
        put_vols = [
            df_tmp[DFKey.ADJ_PUT_1_VOLUME],
            df_tmp[DFKey.ADJ_PUT_2_VOLUME],
            df_tmp[DFKey.ADJ_PUT_3_VOLUME],
            df_tmp[DFKey.ADJ_PUT_4_VOLUME],
            df_tmp[DFKey.ADJ_PUT_5_VOLUME],
        ]

        level1_total = df_tmp[DFKey.ADJ_CALL_1_VOLUME] + df_tmp[DFKey.ADJ_PUT_1_VOLUME]
        all_levels_total = sum(call_vols) + sum(put_vols) + Num.EPS

        volcr = level1_total / all_levels_total

        if self.selected_params["ma_window"] > 1:
            volcr = volcr.rolling(
                window=self.selected_params["ma_window"], min_periods=1
            ).mean()

        df, col = self._add_to_df(df, volcr)

        return df, col
