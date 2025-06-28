from typing import Any

import pandas as pd
from pandas import DataFrame

from ...util.const import DFKey
from ...util.registry import register_geneventfilter
from .geneventfilter import GenEventFilter


@register_geneventfilter("breakout")
class BreakoutFilter(GenEventFilter):
    @property
    def name(self) -> str:
        return "BREAKOUT"

    @property
    def description(self) -> str:
        return """
        Filters rows where 'future_price' breaks above recent highs or
        below recent lows over a lookback window of N candles.
        Commonly used to identify breakout or breakdown events.
        """

    @property
    def params_candidates(self) -> dict[str, list[Any]]:
        return {
            "window": [10, 20, 30],  # lookback period
            "mode": ["breakout", "breakdown", "both"],  # type of breakout
            "strict": [True, False],  # whether equality counts as breakout
        }

    @property
    def name_with_params(self) -> str:
        return f"{self.name}_w{self.selected_params['window']}_{self.selected_params['mode']}"

    @property
    def cols_used(self) -> list[tuple[str, str]]:
        return [DFKey.FUTURE_PRICE_CLOSE]

    def _add_event_mask(self, df: DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        window = self.selected_params["window"]
        mode = self.selected_params["mode"]
        strict = self.selected_params["strict"]

        series = df[DFKey.FUTURE_PRICE_CLOSE]
        prev_high = series.shift(1).rolling(window=window, min_periods=1).max()
        prev_low = series.shift(1).rolling(window=window, min_periods=1).min()

        breakout_cond = series > prev_high if strict else series >= prev_high
        breakdown_cond = series < prev_low if strict else series <= prev_low

        if mode == "breakout":
            mask = breakout_cond
        elif mode == "breakdown":
            mask = breakdown_cond
        elif mode == "both":
            mask = breakout_cond | breakdown_cond
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        df[DFKey.EVENT_MASK] = mask
        return df, mask
