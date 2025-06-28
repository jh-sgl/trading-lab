from typing import Any

import pandas as pd
from pandas import DataFrame

from ...util.const import DFKey
from ...util.registry import register_geneventfilter
from .geneventfilter import GenEventFilter


@register_geneventfilter("extrema")
class ExtremaFilter(GenEventFilter):
    @property
    def name(self) -> str:
        return "EXTREMA"

    @property
    def description(self) -> str:
        return """
        Filters rows where 'future_price' is a local maximum
        or minimum over a past window of N candles.
    """

    @property
    def params_candidates(self) -> dict[str, list[Any]]:
        return {
            "window": [5, 10, 20],
            "mode": ["maxima", "minima", "both"],
            "inclusive": [False, True],  # whether to include equality in comparison
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
        inclusive = self.selected_params["inclusive"]

        series = df[DFKey.FUTURE_PRICE_CLOSE]
        shifted_series = series.shift(1)

        maxima_cond = (
            series >= shifted_series.rolling(window, min_periods=1).max()
            if inclusive
            else series > shifted_series.rolling(window, min_periods=1).max()
        )

        minima_cond = (
            series <= shifted_series.rolling(window, min_periods=1).min()
            if inclusive
            else series < shifted_series.rolling(window, min_periods=1).min()
        )

        if mode == "maxima":
            mask = maxima_cond
        elif mode == "minima":
            mask = minima_cond
        elif mode == "both":
            mask = maxima_cond | minima_cond
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        return df, mask
