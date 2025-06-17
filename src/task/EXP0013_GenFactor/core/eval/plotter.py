import os
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from .statistics import TradingStats


class ResultPlotter:
    def __init__(self, stats: TradingStats, factor_cols: pd.MultiIndex) -> None:
        self._stats = stats
        self._factor_cols = factor_cols

    def _setup_initial_plot(self) -> tuple[Figure, Axes, Axes, Axes, Axes, Axes, Axes]:
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.cla()
        plt.clf()

        fig = plt.figure(figsize=(20, 10))

        timestamp = self._stats.net_stat.timestamp
        plt.suptitle(
            f"{str(timestamp.min())} ~ {str(timestamp.max())}", fontsize=16, y=0.98
        )

        gs = GridSpec(
            2, 4, figure=fig, height_ratios=[1, 0.8], width_ratios=[0.8, 1, 1, 1]
        )

        ax0 = fig.add_subplot(gs[:, 0])
        ax1 = fig.add_subplot(gs[0, 1:3])
        ax2 = fig.add_subplot(gs[1, 1:2])
        ax2b = fig.add_subplot(gs[1, 2:3])
        ax3 = fig.add_subplot(gs[0, 3])
        ax3b = fig.add_subplot(gs[1, 3])
        return fig, ax0, ax1, ax2, ax2b, ax3, ax3b

    def _draw_balance_subplot(self, ax: Axes) -> None:
        ax.plot(
            self._stats.net_stat.balance.index,
            self._stats.net_stat.balance.values,
            color="black",
            alpha=1,
            label="Balance",
            linewidth=0.8,
        )
        ax.plot(
            self._stats.net_stat.long_only_balance.index,
            self._stats.net_stat.long_only_balance.values,
            color="lightsalmon",
            alpha=0.8,
            label="Long Only Balance",
            linewidth=0.65,
        )
        ax.plot(
            self._stats.net_stat.short_only_balance.index,
            self._stats.net_stat.short_only_balance.values,
            color="cornflowerblue",
            alpha=0.8,
            label="Short Only Balance",
            linewidth=0.65,
        )

        ax.plot(
            self._stats.raw_stat.balance.index,
            self._stats.raw_stat.balance.values,
            color="black",
            alpha=0.5,
            label="Balance (Raw)",
            linestyle="dotted",
            linewidth=0.8,
        )
        ax.plot(
            self._stats.raw_stat.long_only_balance.index,
            self._stats.raw_stat.long_only_balance.values,
            color="lightsalmon",
            alpha=0.5,
            label="Long Only Balance (Raw)",
            linestyle="dotted",
            linewidth=0.65,
        )
        ax.plot(
            self._stats.raw_stat.short_only_balance.index,
            self._stats.raw_stat.short_only_balance.values,
            color="cornflowerblue",
            alpha=0.5,
            label="Short Only Balance (Raw)",
            linestyle="dotted",
            linewidth=0.65,
        )

        ax_right = ax.twinx()

        ax_right.bar(
            x=self._stats.net_stat.dailymax_position_signed.index,
            height=self._stats.net_stat.dailymax_position_signed,
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
        self._stats.net_stat.dailymax_position_signed[
            self._stats.net_stat.dailymax_position_signed != 0
        ].hist(bins=200, ax=ax, color="dimgray", alpha=0.7, label="daily max position")
        ax.legend(loc="upper left")
        ax.set_title("Max Position Size Histogram")

        margin_top_5 = self._stats.net_stat.account.sort_values(
            "dailymax_margin", ascending=False
        ).iloc[:5][["dailymax_margin", "dailymax_position", "initial_margin_rate"]]
        position_top_5 = self._stats.net_stat.account.sort_values(
            "dailymax_position", ascending=False
        ).iloc[:5][["dailymax_margin", "dailymax_position", "initial_margin_rate"]]

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
            self._stats.net_stat.total_profit_uwon,
            "Total Return",
            "Simple Interest",
            b=self._stats.net_stat.adjusted_balance,
        )
        _add_table_item(
            table_data,
            self._stats.net_stat.annual_profit_uwon,
            "Annual Return",
            "Simple Interest",
            b=self._stats.net_stat.adjusted_balance,
        )
        _add_table_item(
            table_data,
            self._stats.net_stat.adjusted_balance,
            "Realistic Capital",
            "1.2X of max margin",
        )
        _add_table_item(
            table_data,
            self._stats.net_stat.mdd_capital,
            "MDD Capital",
            "Max Margin + MDD",
        )
        _add_table_item(
            table_data,
            self._stats.net_stat.mdd,
            "Max Drawdown",
            "MDD / Realistic Capital",
            b=self._stats.net_stat.adjusted_balance,
        )
        _add_table_item(
            table_data,
            self._stats.net_stat.max_position_size,
            "Max Position Size",
            a_unit="",
            when=str(self._stats.net_stat.max_position_date),
        )
        _add_table_item(
            table_data,
            self._stats.net_stat.max_margin_size,
            "Max Margin",
            when=str(self._stats.net_stat.max_margin_date),
        )
        _add_table_item(
            table_data,
            self._stats.net_stat.market_exposure,
            "Market Exposure",
            "(Trading days) / (Total days)",
            a_unit="",
        )
        _add_table_item(
            table_data,
            self._stats.net_stat.o_ratio,
            "O-Ratio (Strategy)",
            a_unit="",
        )
        _add_table_item(
            table_data,
            self._stats.net_stat.sharpe_ratio,
            "Sharpe-Ratio (Strategy)",
            a_unit="",
        )
        _add_table_item(table_data, self._stats.net_stat.o_ratio, "O-Ratio", a_unit="")
        _add_table_item(
            table_data,
            self._stats.net_stat.sharpe_ratio,
            "Sharpe-Ratio",
            a_unit="",
        )
        _add_table_item(
            table_data, self._stats.net_stat.r_squared, "R-squared", a_unit=""
        )
        _add_table_item(
            table_data, self._stats.net_stat.skewness, "Skewness", a_unit=""
        )
        _add_table_item(
            table_data, self._stats.net_stat.kurtosis, "Kurtosis", a_unit=""
        )
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

        if len(self._stats.net_stat.drawdown) > 0:
            table_dd = ax_tab.table(
                cellText=self._stats.net_stat.drawdown.iloc[:5].round(3).values,
                colLabels=self._stats.net_stat.drawdown.iloc[:5].columns,
                cellLoc="center",
                loc="center",
            )
            table_dd.auto_set_font_size(False)
            table_dd.set_fontsize(8)
            table_dd.scale(1, 1.5)
        else:
            table_dd = ax_tab.text(
                0.5, 0.5, "No Drawdown Found", fontsize=14, ha="center", va="center"
            )

        ax_tab.axis("off")
        ax_tab.set_title("Worst Draw Downs")

    def _draw_factor_subplot(self, ax: Axes) -> None:
        factor_names = self._factor_cols.get_level_values(1).unique()
        factor_table_data = [[name] for name in sorted(factor_names)]
        factor_table = ax.table(
            cellText=factor_table_data,
            colLabels=["Factor Name"],
            cellLoc="center",
            loc="center",
        )
        factor_table.auto_set_font_size(False)
        factor_table.set_fontsize(7)
        factor_table.scale(1, 1.5)
        ax.axis("off")
        ax.set_title("Factor Names")

    def draw_result(self, save_fp: str | None = None) -> None:
        fig, ax0, ax1, ax2, ax2b, ax3, ax3b = self._setup_initial_plot()
        self._draw_factor_subplot(ax0)
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
