from abc import abstractmethod

import pandas as pd


class Backtestor:
    def __init__(
        self,
        index_multiplier: int = 250_000,
        tick_slippage_size: float = 0.05,
        commision_rate: float = 0.000029,
    ) -> None:
        self._index_multiplier = index_multiplier
        self._tick_slippage_size = tick_slippage_size
        self._commision_rate = commision_rate

    # TODO: type annotation
    @abstractmethod
    def _calc_profit(self) -> None:
        pass

    @abstractmethod
    def run(self, decision_df: pd.DataFrame) -> pd.DataFrame:
        """
        DataFrame Format
        -----------------------------------------
        idx | timestamp | decision | decision_vol
        -----------------------------------------
        """
        pass
