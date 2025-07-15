from typing import Any, Self

import numpy as np
import pandas as pd
import scipy.stats as stats

from ...util.const import MARGIN_RATE, DFKey, Num


class TradingStats:
    class Stats:
        timestamp: Any
        dailymax_position_signed: Any
        balance: Any
        long_only_balance: Any
        short_only_balance: Any
        account: Any
        drawdown: Any
        mdd: Any
        num_trading_days: Any
        total_profit_uwon: Any
        annual_profit_uwon: Any
        o_ratio: Any
        sharpe_ratio: Any
        twa_sharpe_ratio: Any
        skewness: Any
        kurtosis: Any
        r_squared: Any
        adjusted_balance: Any
        market_exposure: Any
        mdd_capital: Any
        max_position_size: Any
        max_position_date: Any
        max_margin_size: Any
        max_margin_date: Any

    def __init__(self, initial_balance: int, balance_adjust_coef: float = 1.2) -> None:
        self.net_stat = self.Stats()
        self.raw_stat = self.Stats()

        self._initial_balance = initial_balance
        self._balance_adjust_coef = balance_adjust_coef

    def _calc_drawdown(
        self, balance: pd.DataFrame, compound: bool = False
    ) -> pd.DataFrame:
        peak = balance.cummax()

        drawdown = peak - balance
        if compound:
            drawdown = drawdown / (1 + peak)

        drawdown = drawdown.fillna(0)

        is_drawdown = drawdown > 0
        is_drawdown_shifted = is_drawdown.shift(fill_value=False)

        start_flag = (~is_drawdown_shifted) & is_drawdown
        drawdown_group = start_flag.cumsum()

        drawdown_group[~is_drawdown] = np.nan

        tmp_df = pd.DataFrame({"drawdown": drawdown, "_group": drawdown_group})
        tmp_df = tmp_df[tmp_df["_group"].notna()]

        if len(tmp_df) == 0:
            drawdown_df = pd.DataFrame(
                columns=["start", "end", "max_drawdown_idx", "max_drawdown_val"]
            )
        else:
            drawdown_df = (
                tmp_df.groupby("_group")
                .apply(
                    lambda group: pd.Series(
                        {
                            "start": group.index[0],
                            "end": group.index[-1],
                            "max_drawdown_idx": group["drawdown"].idxmax(),
                            "max_drawdown_val": group["drawdown"].max(),
                        }
                    )
                )
                .sort_values(by="max_drawdown_val", ascending=False)
                .reset_index(drop=True)
            )

        return drawdown_df

    def _calc_account(self, df: pd.DataFrame) -> pd.DataFrame:
        dailymax_balance = (
            df.groupby(df.index.date)
            .apply(
                lambda x: (
                    x[DFKey.ENTRY_SIGNAL].cumsum()
                    * x[DFKey.PRICE_EXECUTION]
                    * Num.PRICE_MULTIPLIER
                    / Num.UWON
                )
                .abs()
                .max()
            )
            .rename("dailymax_balance")
        )
        # TODO: check difference between 'position_size' of the original codes
        dailymax_position = (
            df.groupby(df.index.date)
            .apply(lambda x: (x[DFKey.ENTRY_SIGNAL].cumsum()).abs().max())
            .rename("dailymax_position")
        )
        # TODO: check integrity
        dailymax_balance_and_position = pd.concat(
            [dailymax_balance, dailymax_position], axis=1
        )
        dailymax_balance_and_position.index = pd.to_datetime(
            dailymax_balance_and_position.index
        )
        # TODO: check integrity / index
        account = pd.merge_asof(
            dailymax_balance_and_position,
            MARGIN_RATE,
            direction="backward",
            left_index=True,
            right_index=True,
        )

        account["dailymax_margin"] = (
            account.dailymax_balance * account.initial_margin_rate
        )
        account["dailymargin_ratio"] = (
            account.dailymax_margin * Num.UWON / self._initial_balance
        )
        return account

    def _calc_dailymax_position_signed(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby(df.index.date, group_keys=False).apply(
            lambda x: np.sign(x[DFKey.ENTRY_SIGNAL].cumsum().iloc[-1])
            * x[DFKey.ENTRY_SIGNAL].cumsum().abs().max()
        )

    def _calc_balance(
        self, df: pd.DataFrame, profit_uwon_key: str
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        balance = df[profit_uwon_key].cumsum()
        long_only_balance = df.loc[df[DFKey.ENTRY_SIGNAL] > 0, profit_uwon_key].cumsum()
        short_only_balance = df.loc[
            df[DFKey.ENTRY_SIGNAL] < 0, profit_uwon_key
        ].cumsum()
        return balance, long_only_balance, short_only_balance

    def _calc_daily_profit_profile(
        self, df: pd.DataFrame, profit_uwon_key: str
    ) -> tuple[float, float, float, float]:
        daily_profit = (
            df[profit_uwon_key].groupby(df.index.date).apply(lambda x: x.sum())
        )
        sharpe_ratio = daily_profit.mean() / daily_profit.std() * (252**0.5)
        skewness = stats.skew(daily_profit)
        kurtosis = stats.kurtosis(daily_profit)
        _, _, r_value, _, _ = stats.linregress(
            np.arange(len(daily_profit)), daily_profit.cumsum().values
        )
        r_squared = r_value**2
        return sharpe_ratio, skewness, kurtosis, r_squared

    def _calc_twa_sharpe_ratio(
        self, pnl_curve: pd.Series, min_len: int = 5000, step: int = 5000
    ):
        """
        Generate trailing sub-curves starting at 0, 100, ..., up to len(curve) - min_len.
        Return time-weighted Sharpe ratio.
        """

        def _sharpe_ratio(returns, periods_per_year=252):
            """Annualized Sharpe ratio (assuming risk-free rate = 0)."""
            if len(returns) < 2:
                return np.nan
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            if std_return == 0:
                return np.nan
            raw_sharpe = mean_return / std_return
            return raw_sharpe * np.sqrt(periods_per_year)

        n = len(pnl_curve)
        scores = []
        weights = []
        for start in range(0, max(n - min_len + 1, 1), step):
            subcurve = pnl_curve[start:]
            returns = np.diff(subcurve)
            score = _sharpe_ratio(returns)
            weight = start + len(subcurve) / 2  # More recent sub-curves get more weight
            scores.append(score)
            weights.append(weight)

        weights = np.array(weights)
        weights = weights / weights.sum()  # normalize
        weighted_score = np.nansum(np.array(scores) * weights)
        return weighted_score

    def calc_stats(self, df: pd.DataFrame, date_range: tuple[str, str]) -> Self:
        time = df.index
        start, end = date_range
        df = df[(start <= time) & (time <= end)].copy()

        for is_costless in [True, False]:
            if is_costless:
                profit_uwon_key = DFKey.RAW_PNL_UWON
                stat = self.raw_stat
            else:
                profit_uwon_key = DFKey.NET_PNL_UWON
                stat = self.net_stat

            stat.timestamp = df.index

            stat.dailymax_position_signed = self._calc_dailymax_position_signed(df)

            stat.balance, stat.long_only_balance, stat.short_only_balance = (
                self._calc_balance(df, profit_uwon_key)
            )
            stat.account = self._calc_account(df)

            stat.drawdown = self._calc_drawdown(stat.balance)
            stat.mdd = stat.drawdown.max_drawdown_val.max()

            stat.num_trading_days = len(set(stat.timestamp.date))
            stat.total_profit_uwon = df[profit_uwon_key].cumsum().iloc[-1]
            stat.annual_profit_uwon = (
                stat.total_profit_uwon / stat.num_trading_days * 252
            )
            stat.o_ratio = stat.annual_profit_uwon / stat.mdd

            stat.sharpe_ratio, stat.skewness, stat.kurtosis, stat.r_squared = (
                self._calc_daily_profit_profile(df, profit_uwon_key)
            )

            stat.twa_sharpe_ratio = self._calc_twa_sharpe_ratio(stat.balance)

            stat.adjusted_balance = (
                stat.account.dailymax_margin.max() * self._balance_adjust_coef
            )

            stat.market_exposure = (
                len(stat.dailymax_position_signed[stat.dailymax_position_signed != 0])
                / stat.num_trading_days
            )

            stat.mdd_capital = stat.account.dailymax_margin.max() + stat.mdd
            stat.max_position_size = stat.account.dailymax_position.max()
            stat.max_position_date = stat.account.dailymax_position.idxmax().date()
            stat.max_margin_size = stat.account.dailymax_margin.max()
            stat.max_margin_date = stat.account.dailymax_margin.idxmax().date()

        return self
