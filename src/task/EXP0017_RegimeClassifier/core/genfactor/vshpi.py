import pandas as pd

from ...util.const import DFKey
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("vshpi")
class VSHPI(GenFactor):
    short_name = "VSHPI"
    full_name = "Vanna-Skew Hedging Pressure Index"
    description = """
        The Vanna-Skew Hedging Pressure Index (VS-HPI) captures the latent hedging flows 
        and directional sentiment shifts implied by changes in option skew,
        open interest positioning**, and volatility smiles, focusing on vanna risk
        and strike-specific sentiment skew.
        It's rooted in how dealers and market makers adjust their hedging
        in response to changes in the underlying price, skew, and volatility surface.

        - Non-linear sentiment flow:
            Changes in skew and OI often signal rebalancing flows
            before they show in price.
        - Dealer positioning:
            Dealers hedging short gamma or vanna exposures must buy/sell
            the underlying as price/volatility shifts.
        - Asymmetry insight:
            Captures when the market is more sensitive to moves
            in one direction due to options' shape.
    """

    params = {
        "ma_window": [1, 20, 50],
        "normalize_iv_gap": [True, False],
        "openint_weight": [0.3, 0.5],
        "skew_weight": [0.3, 0.5, 0.7],
        "vanna_weight": [0.5, 1.0],
    }

    @property
    def name_with_params(self) -> str:
        return (
            f"{self.short_name}_MA{self.selected_params['ma_window']}_"
            f"NIV{int(self.selected_params['normalize_iv_gap'])}_"
            f"OW{int(self.selected_params['openint_weight'] * 10)}_"
            f"SW{int(self.selected_params['skew_weight'] * 10)}_"
            f"VW{int(self.selected_params['vanna_weight'] * 10)}"
        )

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        df_tmp = df.copy()

        # 1. IV skew slopes
        call_skew = (
            df_tmp[DFKey.CALL_IV_2ND_UP_CLOSE] - df_tmp[DFKey.CALL_IV_CLOSE]
        ) / 5
        put_skew = (
            df_tmp[DFKey.PUT_IV_CLOSE] - df_tmp[DFKey.PUT_IV_2ND_DOWN_CLOSE]
        ) / 5
        skew_slope = call_skew - put_skew

        # 2. Open interest skew
        call_oi_skew = (
            df_tmp[DFKey.CALL_OPENINT_2ND_UP_CLOSE] - df_tmp[DFKey.CALL_OPENINT_CLOSE]
        )
        put_oi_skew = (
            df_tmp[DFKey.PUT_OPENINT_CLOSE] - df_tmp[DFKey.PUT_OPENINT_2ND_DOWN_CLOSE]
        )
        oi_skew = call_oi_skew - put_oi_skew

        # 3. Vanna proxy: skew slope × IV gap
        iv_gap = df_tmp[DFKey.CALL_IV_CLOSE] - df_tmp[DFKey.PUT_IV_CLOSE]
        if self.selected_params["normalize_iv_gap"]:
            iv_gap = iv_gap / (
                df_tmp[DFKey.CALL_IV_CLOSE] + df_tmp[DFKey.PUT_IV_CLOSE] + 1e-8
            )
        vanna_pressure = skew_slope * iv_gap

        # Combine into VS-HPI
        df_tmp["vs_hpi_raw"] = (
            self.selected_params["vanna_weight"] * vanna_pressure
            + self.selected_params["skew_weight"] * skew_slope
            + self.selected_params["openint_weight"] * oi_skew
        )

        if self.selected_params["ma_window"] > 1:
            df_tmp["vs_hpi_raw"] = (
                df_tmp["vs_hpi_raw"]
                .rolling(window=self.selected_params["ma_window"], min_periods=1)
                .mean()
            )

        df, col = self._add_to_df(df, df_tmp["vs_hpi_raw"])

        return df, col
