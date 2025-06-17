import asyncio
import logging
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

from task.EXP0012_StrategyEnhancer.core.strategy import Strategy
from util.const import MARGIN_RATE, Num
from util.registry import register_callback

from .const import DFKey


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
                lambda x: (x[DFKey.ENTRY_SIGNAL].cumsum() * x[DFKey.PRICE_EXECUTION] * Num.PRICE_MULTIPLIER / Num.UWON)
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
            lambda x: np.sign(x[DFKey.ENTRY_SIGNAL].cumsum().iloc[-1]) * x[DFKey.ENTRY_SIGNAL].cumsum().abs().max()
        )

    def _calc_balance(self, df: pd.DataFrame, profit_uwon_key: str) -> tuple[pd.Series, pd.Series, pd.Series]:
        balance = df[profit_uwon_key].cumsum()
        long_only_balance = df.loc[df[DFKey.ENTRY_SIGNAL] > 0, profit_uwon_key].cumsum()
        short_only_balance = df.loc[df[DFKey.ENTRY_SIGNAL] < 0, profit_uwon_key].cumsum()
        return balance, long_only_balance, short_only_balance

    def _calc_daily_profit_profile(self, df: pd.DataFrame, profit_uwon_key: str) -> tuple[float, float, float, float]:
        daily_profit = df[profit_uwon_key].groupby(df.index.date).apply(lambda x: x.sum())
        sharpe_ratio = daily_profit.mean() / daily_profit.std() * (252**0.5)
        skewness = stats.skew(daily_profit)
        kurtosis = stats.kurtosis(daily_profit)
        _, _, r_value, _, _ = stats.linregress(np.arange(len(daily_profit)), daily_profit.cumsum().values)
        r_squared = r_value**2
        return sharpe_ratio, skewness, kurtosis, r_squared

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

            stat.balance, stat.long_only_balance, stat.short_only_balance = self._calc_balance(df, profit_uwon_key)
            stat.account = self._calc_account(df)

            stat.drawdown = self._calc_drawdown(stat.balance)
            stat.mdd = stat.drawdown.max_drawdown_val.max()

            stat.num_trading_days = len(set(stat.timestamp.date))
            stat.total_profit_uwon = df[profit_uwon_key].cumsum().iloc[-1]
            stat.annual_profit_uwon = stat.total_profit_uwon / stat.num_trading_days * 252
            stat.o_ratio = stat.annual_profit_uwon / stat.mdd

            stat.sharpe_ratio, stat.skewness, stat.kurtosis, stat.r_squared = self._calc_daily_profit_profile(
                df, profit_uwon_key
            )

            stat.adjusted_balance = stat.account.dailymax_margin.max() * self._balance_adjust_coef

            stat.market_exposure = (
                len(stat.dailymax_position_signed[stat.dailymax_position_signed != 0]) / stat.num_trading_days
            )

            stat.mdd_capital = stat.account.dailymax_margin.max() + stat.mdd
            stat.max_position_size = stat.account.dailymax_position.max()
            stat.max_position_date = stat.account.dailymax_position.idxmax().date()
            stat.max_margin_size = stat.account.dailymax_margin.max()
            stat.max_margin_date = stat.account.dailymax_margin.idxmax().date()

        return self


@register_callback("basic")
class BacktesterCallback(L.Callback):
    def __init__(self, initial_balance: int) -> None:
        super().__init__()
        self._initial_balance = initial_balance
        self._strategy_stats = {"TRAIN": None, "VALID": None, "TOTAL": None}

    def on_validation_end(self, trainer: L.Trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return

        strategy = trainer.datamodule.val_strategy  # val_strategy includes all dates
        date_ranges = {
            "TOTAL": trainer.datamodule.val_date_range,
            "TRAIN": trainer.datamodule.train_date_range,
            "VALID": (trainer.datamodule.train_date_range[1], trainer.datamodule.val_date_range[1]),
        }

        model_signal = pd.Series(0, index=strategy.df.index)
        pred_mask = strategy.df[DFKey.OUTPUT_PRED].notna()
        model_signal[pred_mask] = strategy.df.loc[pred_mask, DFKey.OUTPUT_PRED].apply(lambda x: np.argmax(x) - 1)
        # model_signal[(strategy.df[DFKey.ORIG_SIGNAL] != model_signal)] = 0
        strategy.update_model_signal(model_signal)
        model_df = strategy.backtest(signal_type="model")

        if model_df is None:
            logging.info("No trades found. Skip backtesting.")
            return

        for prefix, date_range in date_ranges.items():
            strategy_stats = self._maybe_calc_strategy_stats(strategy.df, date_range, prefix)

            model_stats = TradingStats(self._initial_balance).calc_stats(model_df, date_range)
            plot_save_fp, backtest_dataframe_fp = self._setup_save(
                trainer.log_dir, prefix, trainer.current_epoch, model_stats
            )

            plotter = ResultPlotter(strategy_stats, model_stats)
            plotter.draw_result(save_fp=plot_save_fp)

    def _setup_save(
        self, trainer_log_dir: str, save_prefix: str, current_epoch: int, model_stats: TradingStats
    ) -> tuple[str, str]:
        plot_save_dir = f"{trainer_log_dir}/plot/E{current_epoch:04d}"
        df_save_dir = f"{trainer_log_dir}/backtest_dataframe/E{current_epoch:04d}"
        os.makedirs(plot_save_dir, exist_ok=True)
        os.makedirs(df_save_dir, exist_ok=True)
        save_filename = f"[{save_prefix}]_O{model_stats.net_stat.o_ratio:.3f}_S{model_stats.net_stat.sharpe_ratio:.3f}"
        plot_save_fp = f"{plot_save_dir}/{save_filename}" + ".png"
        backtest_dataframe_fp = f"{df_save_dir}/{save_filename}" + ".parquet"
        return plot_save_fp, backtest_dataframe_fp

    def _maybe_calc_strategy_stats(
        self, strategy_df: pd.DataFrame, date_range: tuple[str, str], prefix: str
    ) -> TradingStats:
        if self._strategy_stats[prefix] is None:
            self._strategy_stats[prefix] = TradingStats(self._initial_balance).calc_stats(strategy_df, date_range)
        return self._strategy_stats[prefix]


class ResultPlotter:
    def __init__(self, strategy_stats: TradingStats, model_stats: TradingStats) -> None:
        self._strategy_stats = strategy_stats
        self._model_stats = model_stats

    def _setup_initial_plot(self) -> tuple[Figure, Axes, Axes, Axes, Axes, Axes]:
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.cla()
        plt.clf()

        fig = plt.figure(figsize=(20, 10))

        timestamp = self._model_stats.net_stat.timestamp
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
        # model net stat
        ax.plot(
            self._model_stats.net_stat.balance.index,
            self._model_stats.net_stat.balance.values,
            color="black",
            alpha=1,
            label="Balance",
        )
        ax.plot(
            self._model_stats.net_stat.long_only_balance.index,
            self._model_stats.net_stat.long_only_balance.values,
            color="black",
            alpha=1,
            label="Long Only Balance",
            linestyle="dotted",
        )
        ax.plot(
            self._model_stats.net_stat.short_only_balance.index,
            self._model_stats.net_stat.short_only_balance.values,
            color="black",
            alpha=1,
            label="Short Only Balance",
            linestyle="dashed",
        )

        ax.plot(
            self._model_stats.raw_stat.balance.index,
            self._model_stats.raw_stat.balance.values,
            color="black",
            alpha=0.25,
            label="Balance (Raw)",
        )
        ax.plot(
            self._model_stats.raw_stat.long_only_balance.index,
            self._model_stats.raw_stat.long_only_balance.values,
            color="black",
            alpha=0.25,
            label="Long Only Balance (Raw)",
            linestyle="dotted",
        )
        ax.plot(
            self._model_stats.raw_stat.short_only_balance.index,
            self._model_stats.raw_stat.short_only_balance.values,
            color="black",
            alpha=0.25,
            label="Short Only Balance (Raw)",
            linestyle="dashed",
        )

        ax.plot(
            self._strategy_stats.net_stat.balance.index,
            self._strategy_stats.net_stat.balance.values,
            color="cornflowerblue",
            alpha=1,
            label="Balance (Strategy)",
        )
        ax.plot(
            self._strategy_stats.net_stat.long_only_balance.index,
            self._strategy_stats.net_stat.long_only_balance.values,
            color="cornflowerblue",
            alpha=1,
            label="Long Only Balance (Strategy)",
            linestyle="dotted",
        )
        ax.plot(
            self._strategy_stats.net_stat.short_only_balance.index,
            self._strategy_stats.net_stat.short_only_balance.values,
            color="cornflowerblue",
            alpha=1,
            label="Short Only Balance (Strategy)",
            linestyle="dashed",
        )

        ax.plot(
            self._strategy_stats.raw_stat.balance.index,
            self._strategy_stats.raw_stat.balance.values,
            color="cornflowerblue",
            alpha=0.25,
            label="Balance (Raw, Strategy)",
        )
        ax.plot(
            self._strategy_stats.raw_stat.long_only_balance.index,
            self._strategy_stats.raw_stat.long_only_balance.values,
            color="cornflowerblue",
            alpha=0.25,
            label="Long Only Balance (Raw, Strategy)",
            linestyle="dotted",
        )
        ax.plot(
            self._strategy_stats.raw_stat.short_only_balance.index,
            self._strategy_stats.raw_stat.short_only_balance.values,
            color="cornflowerblue",
            alpha=0.25,
            label="Short Only Balance (Raw, Strategy)",
            linestyle="dashed",
        )

        ax_right = ax.twinx()

        ax_right.bar(
            x=self._model_stats.net_stat.dailymax_position_signed.index,
            height=self._model_stats.net_stat.dailymax_position_signed,
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
        self._model_stats.net_stat.dailymax_position_signed[
            self._model_stats.net_stat.dailymax_position_signed != 0
        ].hist(bins=200, ax=ax, color="dimgray", alpha=0.7, label="daily max position")
        ax.legend(loc="upper left")
        ax.set_title("Max Position Size Histogram")

        margin_top_5 = self._model_stats.net_stat.account.sort_values("dailymax_margin", ascending=False).iloc[:5][
            ["dailymax_margin", "dailymax_position", "initial_margin_rate"]
        ]
        position_top_5 = self._model_stats.net_stat.account.sort_values("dailymax_position", ascending=False).iloc[:5][
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
            self._model_stats.net_stat.total_profit_uwon,
            "Total Return",
            "Simple Interest",
            b=self._model_stats.net_stat.adjusted_balance,
        )
        _add_table_item(
            table_data,
            self._model_stats.net_stat.annual_profit_uwon,
            "Annual Return",
            "Simple Interest",
            b=self._model_stats.net_stat.adjusted_balance,
        )
        _add_table_item(
            table_data, self._model_stats.net_stat.adjusted_balance, "Realistic Capital", "1.2X of max margin"
        )
        _add_table_item(table_data, self._model_stats.net_stat.mdd_capital, "MDD Capital", "Max Margin + MDD")
        _add_table_item(
            table_data,
            self._model_stats.net_stat.mdd,
            "Max Drawdown",
            "MDD / Realistic Capital",
            b=self._model_stats.net_stat.adjusted_balance,
        )
        _add_table_item(
            table_data,
            self._model_stats.net_stat.max_position_size,
            "Max Position Size",
            a_unit="",
            when=str(self._model_stats.net_stat.max_position_date),
        )
        _add_table_item(
            table_data,
            self._model_stats.net_stat.max_margin_size,
            "Max Margin",
            when=str(self._model_stats.net_stat.max_margin_date),
        )
        _add_table_item(
            table_data,
            self._model_stats.net_stat.market_exposure,
            "Market Exposure",
            "(Trading days) / (Total days)",
            a_unit="",
        )
        _add_table_item(table_data, self._strategy_stats.net_stat.o_ratio, "O-Ratio (Strategy)", a_unit="")
        _add_table_item(table_data, self._strategy_stats.net_stat.sharpe_ratio, "Sharpe-Ratio (Strategy)", a_unit="")
        _add_table_item(table_data, self._model_stats.net_stat.o_ratio, "O-Ratio", a_unit="")
        _add_table_item(table_data, self._model_stats.net_stat.sharpe_ratio, "Sharpe-Ratio", a_unit="")
        _add_table_item(table_data, self._model_stats.net_stat.r_squared, "R-squared", a_unit="")
        _add_table_item(table_data, self._model_stats.net_stat.skewness, "Skewness", a_unit="")
        _add_table_item(table_data, self._model_stats.net_stat.kurtosis, "Kurtosis", a_unit="")
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

        if len(self._model_stats.net_stat.drawdown) > 0:
            table_dd = ax_tab.table(
                cellText=self._model_stats.net_stat.drawdown.iloc[:5].round(3).values,
                colLabels=self._model_stats.net_stat.drawdown.iloc[:5].columns,
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
