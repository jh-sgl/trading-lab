import pandas as pd

from ...util.const import DFKey
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("participant_corr")
class ParticipantCorrelation(GenFactor):
    short_name = "PCORR"
    full_name = "Participant Flow Correlation"
    description = """
        Computes rolling correlation between price momentum and participant net trading flows.
        Indicates how strongly each participant group is aligned with recent price trends.
    """

    params = {
        "momentum_window": [10, 20, 40, 60],
        "participant": ["foreign", "institutional", "retail"],
    }

    @property
    def name_with_params(self) -> str:
        sp = self.selected_params
        return (
            f"{self.short_name}_{sp['participant'].upper()}_MW{sp['momentum_window']}"
        )

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        sp = self.selected_params
        mw, participant = sp["momentum_window"], sp["participant"]

        momentum = df[DFKey.FUTURE_PRICE_CLOSE].diff(mw)

        flow_col = {
            "foreign": DFKey.FOREIGN_TRADE_CLOSE,
            "institutional": DFKey.INSTITUTIONAL_TRADE_CLOSE,
            "retail": DFKey.INDIVIDUAL_TRADE_CLOSE,
        }[participant]

        flow = df[flow_col].mask(df[flow_col] == 0).ffill().fillna(0)
        correlation = momentum.rolling(mw, min_periods=1).corr(flow)

        df, col = self._add_to_df(df, correlation)
        return df, col
