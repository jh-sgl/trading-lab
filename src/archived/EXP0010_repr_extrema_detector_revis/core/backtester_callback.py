import asyncio
import os
from itertools import product
from typing import Any, Self

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from scipy import stats

from util.const import MARGIN_RATE, Num
from util.registry import register_callback

from .const import DFKey


class TradingStats:
    def __init__(self, price_multiplier: int, initial_balance: int, balance_adjust_coef: float = 1.2) -> None:
        self._price_multiplier = price_multiplier
        self._initial_balance = initial_balance
        self._balance_adjust_coef = balance_adjust_coef

        self.timestamp: pd.DatetimeIndex
        self.balance: Any
        self.long_only_balance: Any
        self.short_only_balance: Any
        self.dailymax_position_signed: Any
        self.account: pd.DataFrame
        self.drawdown_simple: pd.DataFrame
        self.drawdown_compound: pd.DataFrame
        self.mdd: float

        # TODO: add more attributes

    def _calc_drawdown(self, balance: pd.DataFrame, compound: bool = False) -> pd.DataFrame:
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
            drawdown_df = pd.DataFrame(columns=["start", "end", "max_drawdown_idx", "max_drawdown_val"])
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
                lambda x: (x[DFKey.POSITION].cumsum() * x[DFKey.PRICE_ENTER] * self._price_multiplier / Num.UWON)
                .abs()
                .max()
            )
            .rename("dailymax_balance")
        )
        # TODO: check difference between 'position_size' of the original codes
        dailymax_position = (
            df.groupby(df.index.date)
            .apply(lambda x: (x[DFKey.POSITION].cumsum()).abs().max())
            .rename("dailymax_position")
        )
        # TODO: check integrity
        dailymax_balance_and_position = pd.concat([dailymax_balance, dailymax_position], axis=1)
        dailymax_balance_and_position.index = pd.to_datetime(dailymax_balance_and_position.index)
        # TODO: check integrity / index
        account = pd.merge_asof(
            dailymax_balance_and_position, MARGIN_RATE, direction="backward", left_index=True, right_index=True
        )

        account["dailymax_margin"] = account.dailymax_balance * account.initial_margin_rate
        account["dailymargin_ratio"] = account.dailymax_margin * Num.UWON / self._initial_balance
        return account

    def _calc_dailymax_position_signed(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby(df.index.date, group_keys=False).apply(
            lambda x: np.sign(x[DFKey.POSITION].cumsum().iloc[-1]) * x[DFKey.POSITION].cumsum().abs().max()
        )

    def _calc_balance(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
        balance = df[DFKey.PROFIT_UWON].cumsum()
        long_only_balance = df.loc[df[DFKey.POSITION] > 0, DFKey.PROFIT_UWON].cumsum()
        short_only_balance = df.loc[df[DFKey.POSITION] < 0, DFKey.PROFIT_UWON].cumsum()
        return balance, long_only_balance, short_only_balance

    def _calc_daily_profit_profile(self, df: pd.DataFrame) -> tuple[float, float, float, float]:
        daily_profit = df[DFKey.PROFIT_UWON].groupby(df.index.date).apply(lambda x: x.sum())
        sharpe_ratio = daily_profit.mean() / daily_profit.std() * (252**0.5)
        skewness = stats.skew(daily_profit)
        kurtosis = stats.kurtosis(daily_profit)
        _, _, r_value, _, _ = stats.linregress(np.arange(len(daily_profit)), daily_profit.cumsum().values)
        r_squared = r_value**2
        return sharpe_ratio, skewness, kurtosis, r_squared

    # # TODO: type check
    # def _calc_compound_metrics(
    #     self, profit_uwon: pd.Series, adjusted_balance: float, num_trading_days: int
    # ) -> tuple[float, float, float, float]:
    #     return_rate = profit_uwon / adjusted_balance
    #     compound_total_return = (1 + return_rate).prod()
    #     compound_daily_return = ((1 + compound_total_return) ** (1 / num_trading_days)) - 1
    #     compound_annual_return = (1 + compound_daily_return) ** 252 - 1
    #     compound_top_k_drawdown_df = self._calc_drawdown(return_rate, compound=True)
    #     compound_mdd = compound_top_k_drawdown_df.iloc[0].max_drawdown_val
    #     compound_o_ratio = compound_annual_return / compound_mdd
    #     return compound_total_return, compound_annual_return, compound_mdd, compound_o_ratio

    def calc_stats(self, df: pd.DataFrame) -> Self:
        self.timestamp = df.index

        self.dailymax_position_signed = self._calc_dailymax_position_signed(df)

        self.balance, self.long_only_balance, self.short_only_balance = self._calc_balance(df)
        self.account = self._calc_account(df)

        self.drawdown = self._calc_drawdown(self.balance)
        self.mdd = self.drawdown.max_drawdown_val.max()

        self.num_trading_days = len(set(self.timestamp.date))
        self.total_profit_uwon = df[DFKey.PROFIT_UWON].cumsum().iloc[-1]
        self.annual_profit_uwon = self.total_profit_uwon / self.num_trading_days * 252
        self.o_ratio = self.annual_profit_uwon / self.mdd

        self.sharpe_ratio, self.skewness, self.kurtosis, self.r_squared = self._calc_daily_profit_profile(df)

        self.adjusted_balance = self.account.dailymax_margin.max() * self._balance_adjust_coef

        # self.compound_total_return, self.compound_annual_return, self.compound_mdd, self.compound_o_ratio = (
        #     self._calc_compound_metrics(df.profit_uwon, self.adjusted_balance, self.num_trading_days)
        # )
        # self.compound_total_return_percent, self.compound_annual_return_percent, self.compound_mdd_percent = (
        #     self.compound_total_return * 100,
        #     self.compound_annual_return * 100,
        #     self.compound_mdd * 100,
        # )

        self.market_exposure = (
            len(self.dailymax_position_signed[self.dailymax_position_signed != 0]) / self.num_trading_days
        )

        self.mdd_capital = self.account.dailymax_margin.max() + self.mdd
        self.max_position_size = self.account.dailymax_position.max()
        self.max_position_date = self.account.dailymax_position.idxmax().date()
        self.max_margin_size = self.account.dailymax_margin.max()
        self.max_margin_date = self.account.dailymax_margin.idxmax().date()

        return self


@register_callback("basic")
class BacktesterCallback(L.Callback):
    def __init__(
        self,
        tick_slippage_size: float,
        commission_rate: float,
        price_multiplier: int,
        initial_balance: int,
        risk_exposure: float,
        position_scaling_unit: list[float],
        volatility_coef: float,
        position_strategy: str,
        trade_stop_hour: int,
    ) -> None:
        super().__init__()
        self._backtester = Backtester(
            tick_slippage_size,
            commission_rate,
            price_multiplier,
            initial_balance,
            risk_exposure,
            volatility_coef,
            position_strategy,
            trade_stop_hour,
        )
        self._position_scaling_unit = position_scaling_unit
        self._trading_stats = TradingStats(price_multiplier, initial_balance)

    def on_validation_end(self, trainer: L.Trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return

        total_df = trainer.datamodule.val_df  # val_df includes all dates
        total_df = total_df[total_df[DFKey.OUTPUT_PRED].notna()].copy()

        train_only_df = total_df.loc[total_df.index.isin(trainer.datamodule.train_df.index)].copy()
        val_only_df = total_df.loc[~total_df.index.isin(trainer.datamodule.train_df.index)].copy()

        dataset_info = {"TRAIN": train_only_df, "VALID": val_only_df, "TOTAL": total_df}
        ignore_cost_options = [False, True]

        for (prefix, df), ignore_cost, psu in product(
            dataset_info.items(), ignore_cost_options, self._position_scaling_unit
        ):
            prefix = f"{prefix}-COSTLESS" if ignore_cost else prefix
            save_df = False if ignore_cost and prefix == "TOTAL" else True
            self._run(trainer.log_dir, trainer.current_epoch, df, ignore_cost, prefix, psu, save_df)

    def _setup_save(
        self, trainer_log_dir: str, save_prefix: str, current_epoch: int, stats: TradingStats
    ) -> tuple[str, str]:
        plot_save_dir = f"{trainer_log_dir}/plot/E{current_epoch:04d}"
        df_save_dir = f"{trainer_log_dir}/backtest_dataframe/E{current_epoch:04d}"
        os.makedirs(plot_save_dir, exist_ok=True)
        os.makedirs(df_save_dir, exist_ok=True)
        save_filename = f"[{save_prefix}]_O{stats.o_ratio:.3f}_S{stats.sharpe_ratio:.3f}"
        plot_save_fp = f"{plot_save_dir}/{save_filename}" + ".png"
        backtest_dataframe_fp = f"{df_save_dir}/{save_filename}" + ".parquet"
        return plot_save_fp, backtest_dataframe_fp

    def _run(
        self,
        log_dir: str,
        current_epoch: int,
        df: pd.DataFrame,
        ignore_cost: bool,
        save_prefix: str,
        position_scaling_unit: float,
        save_df: bool,
    ) -> None:
        df = self._backtester.run_backtest(df, position_scaling_unit, ignore_cost=ignore_cost)
        stats = self._trading_stats.calc_stats(df)
        plot_save_fp, backtest_dataframe_fp = self._setup_save(
            log_dir, save_prefix + f"-PSU{position_scaling_unit}", current_epoch, stats
        )

        plotter = ResultPlotter(stats)
        plotter.draw_result(save_fp=plot_save_fp)

        async def _ato_parquet(df, backtest_dataframe_fp):
            await asyncio.to_thread(
                df[
                    [
                        DFKey.FUTURE_PRICE_OPEN,
                        DFKey.FUTURE_PRICE_HIGH,
                        DFKey.FUTURE_PRICE_LOW,
                        DFKey.FUTURE_PRICE_CLOSE,
                        DFKey.PRICE_MOVE,
                        DFKey.OUTPUT_PRED,
                        DFKey.POSITION,
                        DFKey.PROFIT,
                        DFKey.COST,
                    ]
                ].to_parquet,
                backtest_dataframe_fp,
            )

        if save_df:
            asyncio.run(_ato_parquet(df, backtest_dataframe_fp))


class Backtester:
    def __init__(
        self,
        tick_slippage_size: float,
        commission_rate: float,
        price_multiplier: float,
        initial_balance: int,
        risk_exposure: float,
        volatility_coef: float,
        position_strategy: str,
        trade_stop_hour: int,
    ) -> None:
        self._tick_slippage_size = tick_slippage_size
        self._commission_rate = commission_rate
        self._price_multiplier = price_multiplier
        self._initial_balance = initial_balance
        self._risk_exposure = risk_exposure
        self._volatility_coef = volatility_coef
        self._position_strategy = position_strategy
        self._trade_stop_hour = trade_stop_hour

    def _calc_max_position(self, volatility: pd.Series) -> pd.Series:
        return (self._initial_balance * self._risk_exposure) / (
            volatility * self._volatility_coef * self._price_multiplier + 1e-10
        )

    def _decide_position(self, df: pd.DataFrame, position_scaling_unit: float) -> pd.Series:
        output_pred = df[DFKey.OUTPUT_PRED]
        volatility = df[DFKey.VOLATILITY_50]

        if self._position_strategy == "risk_volatility_adjustment":
            max_position = self._calc_max_position(volatility.clip(0.1, 10))
            weight = output_pred / (position_scaling_unit + 1e-15)
            weight = np.sign(weight) * np.sqrt(abs(weight))
            position = (weight * max_position).apply(np.trunc)
            position[abs(output_pred) <= position_scaling_unit] = 0
            position = position.clip(-50, 50)

        elif self._position_strategy == "thresholding":
            position = np.sign(output_pred) * (abs(output_pred) >= position_scaling_unit).astype(int)

        elif self._position_strategy == "naive":
            position = pd.Series(0, index=output_pred.index)
            position[output_pred > 0] = 1
            position[output_pred < 0] = -1

        elif self._position_strategy == "classification":
            max_position = self._calc_max_position(volatility.clip(0.1, 10))
            position = np.round((output_pred * max_position).clip(1, float("inf")), 0) * np.sign(output_pred)

        position[position.index.hour >= self._trade_stop_hour] = 0

        return position

    def _calc_slippage_cost(self, position: pd.Series) -> pd.Series:
        return abs(position) * self._tick_slippage_size * 2

    def _calc_commision_cost(self, position: pd.Series, enter_price: pd.Series) -> pd.Series:
        return abs(position) * enter_price * self._commission_rate * 2

    def _calc_contract_cost(self, position: pd.Series, enter_price: pd.Series, ignore_cost: bool) -> pd.Series:
        if not ignore_cost:
            return self._calc_slippage_cost(position) + self._calc_commision_cost(position, enter_price)
        else:
            return pd.Series(0, index=position.index)

    def _calc_profit(self, df: pd.DataFrame, apply_price_multiplier: bool) -> pd.Series:
        profit = df[DFKey.PRICE_MOVE] * df[DFKey.POSITION] - df[DFKey.COST]

        if apply_price_multiplier:
            profit *= self._price_multiplier

        return profit

    def run_backtest(self, df: pd.DataFrame, position_scaling_unit: float, ignore_cost: bool = False) -> pd.DataFrame:
        df[DFKey.POSITION] = self._decide_position(df, position_scaling_unit)
        df[DFKey.COST] = self._calc_contract_cost(df[DFKey.POSITION], df[DFKey.PRICE_ENTER], ignore_cost)
        df[DFKey.PROFIT] = self._calc_profit(df, apply_price_multiplier=True)
        df[DFKey.PROFIT_UWON] = df[DFKey.PROFIT] / Num.UWON
        return df


class ResultPlotter:
    def __init__(self, stats: TradingStats) -> None:
        # TODO: validate stats? with pydantic? or dataclasses?
        self._stats = stats

    def _setup_initial_plot(self) -> tuple[Figure, Axes, Axes, Axes, Axes, Axes]:
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.cla()
        plt.clf()

        fig = plt.figure(figsize=(20, 10))

        timestamp = self._stats.timestamp
        plt.suptitle(f"{str(timestamp.min())} ~ {str(timestamp.max())}", fontsize=16, y=0.98)

        num_cols = 3
        gs = GridSpec(2, num_cols, figure=fig, height_ratios=[1, 0.8])

        ax1 = fig.add_subplot(gs[0, :2])
        ax2 = fig.add_subplot(gs[1, :1])
        ax2b = fig.add_subplot(gs[1, 1:2])
        ax3 = fig.add_subplot(gs[0, 2])
        ax3b = fig.add_subplot(gs[1, 2])
        return fig, ax1, ax2, ax2b, ax3, ax3b

    def _draw_balance_subplot(self, ax: Axes) -> None:
        ax.plot(
            self._stats.balance.index,
            self._stats.balance.values,
            color="indianred",
            alpha=1,
            label="Balance",
        )
        ax.plot(
            self._stats.long_only_balance.index,
            self._stats.long_only_balance.values,
            color="lightsalmon",
            alpha=0.5,
            label="Long Only Balance",
        )
        ax.plot(
            self._stats.short_only_balance.index,
            self._stats.short_only_balance.values,
            color="cornflowerblue",
            alpha=0.5,
            label="Short Only Balance",
        )

        ax_right = ax.twinx()

        ax_right.bar(
            x=self._stats.dailymax_position_signed.index,
            height=self._stats.dailymax_position_signed,
            color="dimgray",
            alpha=0.5,
            label="Position Size",
        )

        ax.legend(loc="upper left")
        ax_right.legend(loc="lower right")

        ax.set_ylabel("Balance")
        ax_right.set_ylabel("Position Size")

        ax.set_title(f"Balance and Position Size Over Time")

    def _draw_position_subplot(self, ax: Axes, ax_tab: Axes) -> None:
        self._stats.dailymax_position_signed[self._stats.dailymax_position_signed != 0].hist(
            bins=200, ax=ax, color="dimgray", alpha=0.7, label="daily max position"
        )
        ax.legend(loc="upper left")
        ax.set_title("Max Position Size Histogram")

        margin_top_5 = self._stats.account.sort_values("dailymax_margin", ascending=False).iloc[:5][
            ["dailymax_margin", "dailymax_position", "initial_margin_rate"]
        ]
        position_top_5 = self._stats.account.sort_values("dailymax_position", ascending=False).iloc[:5][
            ["dailymax_margin", "dailymax_position", "initial_margin_rate"]
        ]

        margin_position_rank = pd.concat([margin_top_5, position_top_5], axis=0)
        margin_position_rank.index = margin_position_rank.index.date
        margin_position_rank = margin_position_rank.reset_index()

        margin_table = ax_tab.table(
            cellText=margin_position_rank.round(3).values,
            colLabels=margin_position_rank.columns,
            cellLoc="center",
            loc="center",
        )
        margin_table.auto_set_font_size(False)
        margin_table.set_fontsize(7)
        margin_table.scale(1, 1.5)
        ax_tab.axis("off")
        ax_tab.set_title("Max Position size")

    def _draw_statistics_subplot(self, ax: Axes, ax_tab: Axes) -> None:
        def _add_table_item(
            table_data: list[Any],
            a: float,
            metric_name: str,
            desc: str = "",
            a_unit: str = "UWON",
            b: float | None = None,
            when: str | None = None,
        ) -> list[Any]:
            value_str = f"{a:.2f} {a_unit}"
            if b is not None:
                value_str += f" {a / b * 100:.3f} %"
            if when is not None:
                value_str += f" @ {when}"
            table_data.append([metric_name, value_str, desc])
            return table_data

        table_data = []
        _add_table_item(
            table_data,
            self._stats.total_profit_uwon,
            "Total Return",
            "Simple Interest",
            b=self._stats.adjusted_balance,
        )
        _add_table_item(
            table_data,
            self._stats.annual_profit_uwon,
            "Annual Return",
            "Simple Interest",
            b=self._stats.adjusted_balance,
        )
        _add_table_item(table_data, self._stats.adjusted_balance, "Realistic Capital", "1.2X of max margin")
        _add_table_item(table_data, self._stats.mdd_capital, "MDD Capital", "Max Margin + MDD")
        _add_table_item(
            table_data, self._stats.mdd, "Max Drawdown", "MDD / Realistic Capital", b=self._stats.adjusted_balance
        )
        _add_table_item(
            table_data,
            self._stats.max_position_size,
            "Max Position Size",
            a_unit="",
            when=str(self._stats.max_position_date),
        )
        _add_table_item(
            table_data,
            self._stats.max_margin_size,
            "Max Margin",
            when=str(self._stats.max_margin_date),
        )
        _add_table_item(
            table_data, self._stats.market_exposure, "Market Exposure", "(Trading days) / (Total days)", a_unit=""
        )
        _add_table_item(table_data, self._stats.o_ratio, "O-Ratio", a_unit="")
        _add_table_item(table_data, self._stats.sharpe_ratio, "Sharpe-Ratio", a_unit="")
        _add_table_item(table_data, self._stats.r_squared, "R-squared", a_unit="")
        _add_table_item(table_data, self._stats.skewness, "Skewness", a_unit="")
        _add_table_item(table_data, self._stats.kurtosis, "Kurtosis", a_unit="")
        # _add_table_item(
        #     table_data,
        #     self._stats.compound_total_return_percent,
        #     "Total Return (Compound)",
        #     "Compound Interest",
        #     "%",
        # )
        # _add_table_item(
        #     table_data,
        #     self._stats.compound_annual_return_percent,
        #     "Annual Return (Compound)",
        #     "Compound Interest",
        #     "%",
        # )
        # _add_table_item(
        #     table_data,
        #     self._stats.compound_mdd_percent,
        #     "Max Drawdown (Compound)",
        #     "Compound Interest",
        #     "%",
        # )
        # _add_table_item(
        #     table_data,
        #     self._stats.compound_o_ratio,
        #     "O-Ratio (Compound)",
        #     "Compound Interest",
        #     a_unit="",
        # )
        table = ax.table(
            cellText=table_data,
            colLabels=["Metric", "Value", "Description"],
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1, 1.5)
        ax.axis("off")
        ax.set_title("Performance Metrics", pad=20)

        if len(self._stats.drawdown) > 0:
            table_dd = ax_tab.table(
                cellText=self._stats.drawdown.iloc[:5].round(3).values,
                colLabels=self._stats.drawdown.iloc[:5].columns,
                cellLoc="center",
                loc="center",
            )
            table_dd.auto_set_font_size(False)
            table_dd.set_fontsize(8)
            table_dd.scale(1, 1.5)
        else:
            table_dd = ax_tab.text(0.5, 0.5, "No Drawdown Found", fontsize=14, ha="center", va="center")

        ax_tab.axis("off")
        ax_tab.set_title("Worst Draw Downs")

    def draw_result(self, save_fp: str | None = None) -> None:
        fig, ax1, ax2, ax2b, ax3, ax3b = self._setup_initial_plot()
        self._draw_balance_subplot(ax1)
        self._draw_position_subplot(ax2, ax2b)
        self._draw_statistics_subplot(ax3, ax3b)
        plt.tight_layout()

        if save_fp is not None:
            self.save(fig, save_fp)

        plt.close()

    def save(self, fig: Figure, save_fp: str) -> None:
        os.makedirs(os.path.dirname(save_fp), exist_ok=True)
        fig.savefig(save_fp)
