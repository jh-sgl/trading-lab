import pandas as pd

from ...util.const import DFKey, Num
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("adjo_cdi")
class AdjoCDI(GenFactor):
    short_name = "AdjoCDI"
    full_name = "Adjacent Option-based Intraday Conviction Drain Index"
    description = """
        Detects fading directional conviction throughout each session by combining
        volatility compression, option pulse reversal, and book skew relative to
        initial session movement. Designed for intraday strategies with EOD liquidation.
    """

    params = {
        "option_levels": [1, 3, 5],
        "early_window": [6, 12],
        "ema_span": [1, 3, 6, 12],
    }

    @property
    def name_with_params(self) -> str:
        p = self.selected_params
        return f"{self.short_name}_L{p['option_levels']}_Early{p['early_window']}_EMA{p['ema_span']}"

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        import pandas as pd
        from pandas.api.types import is_datetime64_any_dtype as is_datetime

        df_tmp = df.copy()
        L = self.selected_params["option_levels"]
        W = self.selected_params["early_window"]
        ema = self.selected_params["ema_span"]
        eps = Num.EPS

        # Ensure datetime index or column
        if "datetime" in df_tmp.columns:
            dt_index = df_tmp["datetime"]
        else:
            dt_index = df_tmp.index

        if not is_datetime(dt_index):
            dt_index = pd.to_datetime(dt_index)

        df_tmp["date"] = dt_index.date

        results = []

        for d, group in df_tmp.groupby("date"):
            g = group.copy()
            if g.empty:
                continue

            # Session open
            session_open = g[DFKey.FUTURE_PRICE_OPEN].iloc[0]
            push_pct = (g[DFKey.FUTURE_PRICE_CLOSE] - session_open) / (
                session_open + eps
            )

            # Volatility drain
            hl = g[DFKey.FUTURE_PRICE_HIGH] - g[DFKey.FUTURE_PRICE_LOW]
            short_vol = hl.rolling(window=3, min_periods=1).mean()
            long_vol = hl.rolling(window=12, min_periods=1).mean()
            vol_drain = short_vol / (long_vol + eps)

            # Option pulse reversal
            call_pulse = pd.concat(
                [
                    g[getattr(DFKey, f"ADJ_CALL_{i}_PRICE_CLOSE")]
                    - g[getattr(DFKey, f"ADJ_CALL_{i}_PRICE_OPEN")]
                    for i in range(1, L + 1)
                ],
                axis=1,
            ).sum(axis=1)

            put_pulse = pd.concat(
                [
                    g[getattr(DFKey, f"ADJ_PUT_{i}_PRICE_CLOSE")]
                    - g[getattr(DFKey, f"ADJ_PUT_{i}_PRICE_OPEN")]
                    for i in range(1, L + 1)
                ],
                axis=1,
            ).sum(axis=1)

            pulse_now = call_pulse - put_pulse
            pulse_early = pulse_now.rolling(window=W, min_periods=1).mean()
            pulse_reversal = -1 * (pulse_now - pulse_early)

            # Orderbook skew (L1 vs L-level)
            ob_l1 = (
                g[getattr(DFKey, "BUY_ORDER_1_QTY_CLOSE")]
                + g[getattr(DFKey, "SELL_ORDER_1_QTY_CLOSE")]
            )
            ob_ln = (
                g[getattr(DFKey, f"BUY_ORDER_{L}_QTY_CLOSE")]
                + g[getattr(DFKey, f"SELL_ORDER_{L}_QTY_CLOSE")]
            )
            ob_skew = ob_l1 - ob_ln

            # Combine
            signal = pulse_reversal * vol_drain * ob_skew * push_pct

            if ema > 1:
                signal = signal.ewm(span=ema, min_periods=1).mean()

            results.append(signal)

        # Merge results across all dates
        final_signal = pd.concat(results).sort_index()
        df, col = self._add_to_df(df_tmp, final_signal)

        return df, col
