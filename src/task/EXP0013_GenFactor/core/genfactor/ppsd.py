import pandas as pd

from ...util.const import DFKey
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("ppsd")
class PPSD(GenFactor):
    short_name = "PPSD"
    full_name = "Participant Price-Sensitivity Divergence"
    description = """
        PPSD measures divergence in price-responsiveness between retail and institutional/foreign participants.
        It computes the rolling correlation between short-term futures price momentum and each participant group's net trading volume.

        - Captures directional conviction by smart money vs reactive behavior from retail
        - Positive PPSD: institutions and foreigners align with price momentum, retail lags or resists
        - Negative PPSD: retail follows momentum, smart money resists or fades
        - Orthogonal to orderbook, volatility, pivot, and skew-based features
    """

    params = {
        "momentum_window": [1, 5, 10, 20],
        "correlation_window": [12, 24, 48],
    }

    @property
    def name_with_params(self) -> str:
        sp = self.selected_params
        return (
            f"{self.short_name}_MW{sp['momentum_window']}_CW{sp['correlation_window']}"
        )

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        df_tmp = df.copy()
        sp = self.selected_params
        mw = sp["momentum_window"]
        cw = sp["correlation_window"]

        # Price momentum (ΔP)
        price_momentum = df_tmp[DFKey.FUTURE_PRICE_CLOSE].diff(mw)

        # Participant net trades
        foreign = df_tmp[DFKey.FOREIGN_TRADE_CLOSE]
        institutional = df_tmp[DFKey.INSTITUTIONAL_TRADE_CLOSE]
        retail = df_tmp[DFKey.INDIVIDUAL_TRADE_CLOSE]

        # Fill 0 values at the first row of each day
        foreign = foreign.mask(foreign == 0).ffill().fillna(0)
        institutional = institutional.mask(institutional == 0).ffill().fillna(0)
        retail = retail.mask(retail == 0).ffill().fillna(0)

        # Rolling correlation of price momentum with each participant flow
        rho_foreign = price_momentum.rolling(cw).corr(foreign)
        rho_institutional = price_momentum.rolling(cw).corr(institutional)
        rho_retail = price_momentum.rolling(cw).corr(retail)

        # Compute divergence
        ppsd_raw = (rho_foreign + rho_institutional) - rho_retail

        # Add to DataFrame
        df, col = self._add_to_df(df, ppsd_raw)
        return df, col
