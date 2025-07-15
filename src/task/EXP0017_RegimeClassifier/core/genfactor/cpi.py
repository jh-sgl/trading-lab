import pandas as pd
import numpy as np

from ...util.const import DFKey
from ...util.registry import register_genfactor
from .gen_factor import GenFactor


@register_genfactor("cpi")
class CPI(GenFactor):
    short_name = "CPI"
    full_name = "Charm Pressure Index"
    description = """
        The Charm Pressure Index (CPI) estimates time-decay-induced delta hedging pressure 
        from options, by combining a proxy for charm (∂Δ/∂τ) and open interest asymmetry.
        
        - Charm is approximated using IV and time-to-expiry, auto-detected by calendar rules.
        - Positive CPI: net buy-side hedging pressure.
        - Negative CPI: net sell-side hedging pressure.
    """

    params = {
        "smoothing_window": [1, 5, 10],
    }

    @property
    def name_with_params(self) -> str:
        return f"{self.short_name}_S{self.selected_params['smoothing_window']}"

    @staticmethod
    def get_kospi_option_expiry_dates(df_dates: pd.Series) -> pd.Series:
        """Return a series of expiry dates aligned to df_dates (2nd Thursday rule)."""
        from pandas.tseries.offsets import BDay

        expiry_map = {}
        unique_months = df_dates.dt.to_period("M").unique()

        for month in unique_months:
            month_start = pd.Timestamp(str(month))
            thursdays = pd.date_range(
                start=month_start,
                end=month_start + pd.offsets.MonthEnd(0),
                freq="W-THU",
            )

            if len(thursdays) < 2:
                continue  # just for safety

            second_thursday = thursdays[1]
            # If holiday or non-trading, fallback to previous business day
            expiry = second_thursday
            while expiry not in df_dates.values and expiry >= thursdays[0]:
                expiry -= BDay(1)
            expiry_map[str(month)] = expiry

        expiry_series = df_dates.dt.to_period("M").astype(str).map(expiry_map)
        return pd.to_datetime(expiry_series)

    def add_genfactor(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
        df_tmp = df.copy()
        smoothing = self.selected_params["smoothing_window"]

        date_dt = pd.to_datetime(df_tmp[DFKey.DATE])

        # Compute tau from expiry rule
        df_tmp["expiry_date"] = self.get_kospi_option_expiry_dates(date_dt)
        df_tmp["tau_days"] = (df_tmp["expiry_date"] - date_dt).dt.days.clip(lower=1)
        tau = df_tmp["tau_days"] / 252.0  # annualized
        sqrt_tau = np.sqrt(tau)

        # Charm proxy
        df_tmp["charm_call"] = (
            -df_tmp[DFKey.CALL_IV_CLOSE]
            / (2 * sqrt_tau)
            * df_tmp[DFKey.CALL_OPENINT_CLOSE]
        )
        df_tmp["charm_put"] = (
            -df_tmp[DFKey.PUT_IV_CLOSE]
            / (2 * sqrt_tau)
            * df_tmp[DFKey.PUT_OPENINT_CLOSE]
        )

        # Net directional charm pressure
        df_tmp["cpi_raw"] = df_tmp["charm_call"] - df_tmp["charm_put"]

        if smoothing > 1:
            df_tmp["cpi_raw"] = (
                df_tmp["cpi_raw"].rolling(window=smoothing, min_periods=1).mean()
            )

        df, col = self._add_to_df(df, df_tmp["cpi_raw"])
        return df, col
