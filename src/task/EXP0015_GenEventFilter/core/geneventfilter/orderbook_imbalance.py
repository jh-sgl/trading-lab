from typing import Any

import pandas as pd

from ...util.const import DFKey, Num
from ...util.registry import register_geneventfilter
from .geneventfilter import GenEventFilter


@register_geneventfilter("orderbook_imbalance")
class OrderbookImbalanceFilter(GenEventFilter):
    @property
    def name(self) -> str:
        return "ORDERBOOK_IMBALANCE"

    @property
    def description(self) -> str:
        return """
        Filters rows where the aggregated BUY vs SELL order quantities at top N levels
        are imbalanced beyond a ratio threshold. Helps identify supply/demand pressure.
        """

    @property
    def params_candidates(self) -> dict[str, list[Any]]:
        return {
            "levels": [1, 3, 5],  # how many levels of the book to consider
            "threshold": [1.5, 2.0, 3.0],  # imbalance ratio
            "direction": ["buy", "sell", "both"],  # which direction to include
        }

    @property
    def name_with_params(self) -> str:
        return f"{self.name}_l{self.selected_params['levels']}_t{self.selected_params['threshold']}_{self.selected_params['direction']}"

    @property
    def cols_used(self) -> list[tuple[str, str]]:
        cols = []
        for lvl in range(1, self.selected_params["levels"] + 1):
            cols += [
                getattr(DFKey, f"BUY_ORDER_{lvl}_QTY_CLOSE"),
                getattr(DFKey, f"SELL_ORDER_{lvl}_QTY_CLOSE"),
            ]
        return cols

    def add_event_mask(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
        levels = self.selected_params["levels"]
        threshold = self.selected_params["threshold"]
        direction = self.selected_params["direction"]

        buy_cols = [
            getattr(DFKey, f"BUY_ORDER_{lvl}_QTY_CLOSE") for lvl in range(1, levels + 1)
        ]
        sell_cols = [
            getattr(DFKey, f"SELL_ORDER_{lvl}_QTY_CLOSE")
            for lvl in range(1, levels + 1)
        ]

        buy_sum = df[buy_cols].sum(axis=1)
        sell_sum = df[sell_cols].sum(axis=1)

        ratio = buy_sum / (sell_sum + Num.EPS)  # avoid division by zero

        if direction == "buy":
            mask = ratio > threshold
        elif direction == "sell":
            mask = ratio < 1 / threshold
        elif direction == "both":
            mask = (ratio > threshold) | (ratio < 1 / threshold)
        else:
            raise ValueError(f"Unsupported direction: {direction}")

        df[DFKey.EVENT_MASK] = mask
        return df, self.cols_used
