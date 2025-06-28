from typing import Any

import numpy as np
import pandas as pd

from ...util.const import DFKey
from ...util.registry import register_geneventfilter
from .geneventfilter import GenEventFilter


@register_geneventfilter("volume_surge")
class VolumeSurgeFilter(GenEventFilter):
    @property
    def name(self) -> str:
        return "VOLUME_SURGE"

    @property
    def description(self) -> str:
        return """
        Filters rows where 'FUTURE_VOLUME' exceeds recent rolling average
        by a specified multiple threshold. Useful for identifying abnormal
        trading activity.
        """

    @property
    def params_candidates(self) -> dict[str, list[Any]]:
        return {
            "window": [10, 20, 30],  # lookback window for average volume
            "threshold": [1.5, 2.0, 3.0],  # how many times greater than average
            "mode": ["absolute", "zscore"],  # type of comparison
        }

    @property
    def name_with_params(self) -> str:
        return f"{self.name}_w{self.selected_params['window']}_t{self.selected_params['threshold']}_{self.selected_params['mode']}"

    @property
    def cols_used(self) -> list[tuple[str, str]]:
        return [DFKey.FUTURE_VOLUME]

    def _add_event_mask(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        window = self.selected_params["window"]
        threshold = self.selected_params["threshold"]
        mode = self.selected_params["mode"]

        volume = df[DFKey.FUTURE_VOLUME]

        if mode == "absolute":
            rolling_avg = volume.shift(1).rolling(window, min_periods=1).mean()
            mask = volume > (rolling_avg * threshold)
        elif mode == "zscore":
            rolling_mean = volume.shift(1).rolling(window, min_periods=1).mean()
            rolling_std = volume.shift(1).rolling(window, min_periods=1).std(ddof=0)
            zscore = (volume - rolling_mean) / rolling_std.replace(0, np.nan)
            mask = zscore > threshold
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        return df, mask
