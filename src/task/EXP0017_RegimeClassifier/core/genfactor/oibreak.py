import pandas as pd

from ...util.const import DFKey
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("oibreak")
class OIBreak(GenFactor):
    short_name = "OIBreak"
    full_name = "Generalized Option-Implied Breakout Index"
    description = """
        Measures whether the option market is pricing moves beyond the actual 5-minute intrabar range,
        using CALL and PUT premiums at N nearest strikes (levels = 1 to N).

        For each level n in 1 to levels:
        - CALL_n_PRICE is compared to (HIGH - CLOSE)
        - PUT_n_PRICE is compared to (CLOSE - LOW)

        Then:
        - UpsideGap_n = CALL_n_PRICE - max(0, HIGH - CLOSE)
        - DownsideGap_n = PUT_n_PRICE - max(0, CLOSE - LOW)

        OIBreak = mean(UpsideGap_n - DownsideGap_n for n in 1..N)

        Interpretation:
        - Positive → bullish breakout expectation
        - Negative → bearish breakdown expectation
        - Near 0 → options reflect current realized range

        Parameters:
        - levels: Number of adjacent strikes to include (1–5)
        - ma_window: Smoothing window for stability
    """

    params = {
        "levels": [1, 2, 3, 4, 5],
        "ma_window": [1, 3, 5, 10, 30],
    }

    @property
    def name_with_params(self) -> str:
        return f"{self.short_name}_L{self.selected_params['levels']}_MA{self.selected_params['ma_window']}"

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        df_tmp = df.copy()
        high = df_tmp[DFKey.FUTURE_PRICE_HIGH]
        low = df_tmp[DFKey.FUTURE_PRICE_LOW]
        close = df_tmp[DFKey.FUTURE_PRICE_CLOSE]

        N = self.selected_params["levels"]
        upside_gaps = []
        downside_gaps = []

        for n in range(1, N + 1):
            call = df_tmp[getattr(DFKey, f"ADJ_CALL_{n}_PRICE_CLOSE")]
            put = df_tmp[getattr(DFKey, f"ADJ_PUT_{n}_PRICE_CLOSE")]

            gap_up = call - (high - close).clip(lower=0)
            gap_dn = put - (close - low).clip(lower=0)

            upside_gaps.append(gap_up)
            downside_gaps.append(gap_dn)

        breakout_index = (sum(upside_gaps) - sum(downside_gaps)) / N

        if self.selected_params["ma_window"] > 1:
            breakout_index = breakout_index.rolling(
                window=self.selected_params["ma_window"], min_periods=1
            ).mean()

        df, col = self._add_to_df(df, breakout_index)

        return df, col
