import os
import shutil

import lightning as L
from omegaconf import DictConfig

from ...util.registry import build_backtester, register_callback
from .plotter import ResultPlotter
from .statistics import TradingStats


@register_callback("backtester")
class BacktesterCallback(L.Callback):
    def __init__(self, initial_balance: int, backtester: DictConfig) -> None:
        super().__init__()
        self._initial_balance = initial_balance
        self._backtester = build_backtester(backtester)
        self._stats = {"TRAIN": None, "OOS": None, "TOTAL": None}

    def on_validation_end(self, trainer: L.Trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return

        df = trainer.datamodule.df
        date_ranges = {
            "TOTAL": trainer.datamodule.total_date_range,
            "TRAIN": trainer.datamodule.train_date_range,
            "VALID": (trainer.datamodule.train_date_range[1], "2021-12-31"),
            "TEST": ("2022-01-01", trainer.datamodule.total_date_range[1]),
            "OOS": (
                trainer.datamodule.train_date_range[1],
                trainer.datamodule.total_date_range[1],
            ),
        }
        df = self._backtester.backtest_with_numba(df)
        if df is None:
            return

        for prefix, date_range in date_ranges.items():
            stats = TradingStats(self._initial_balance).calc_stats(df, date_range)
            plotter = ResultPlotter(stats, trainer.datamodule.factor_cols)

            plot_save_fp, backtest_dataframe_fp = self._setup_save(
                trainer.log_dir, prefix, trainer.current_epoch, stats
            )

            if prefix == "TOTAL":
                total_sharpe = stats.net_stat.sharpe_ratio
                plotter.draw_result(save_fp=plot_save_fp)
                df.to_parquet(backtest_dataframe_fp)
            elif prefix == "OOS":
                oos_sharpe = stats.net_stat.sharpe_ratio
            elif prefix == "TRAIN":
                train_sharpe = stats.net_stat.sharpe_ratio
            elif prefix == "VALID":
                valid_sharpe = stats.net_stat.sharpe_ratio
            elif prefix == "TEST":
                test_sharpe = stats.net_stat.sharpe_ratio

        shutil.move(
            f"{trainer.log_dir}/plot/E{trainer.current_epoch:04d}",
            f"{trainer.log_dir}/plot/E{trainer.current_epoch:04d}_TR{train_sharpe:.3f}_VAL{valid_sharpe:.3f}_TEST{test_sharpe:.3f}_OOS{oos_sharpe:.3f}_TOT{total_sharpe:.3f}",
        )

    def _setup_save(
        self,
        trainer_log_dir: str,
        save_prefix: str,
        current_epoch: int,
        stats: TradingStats,
    ) -> tuple[str, str]:
        plot_save_dir = f"{trainer_log_dir}/plot/E{current_epoch:04d}"
        df_save_dir = f"{trainer_log_dir}/backtest_dataframe/E{current_epoch:04d}"
        os.makedirs(plot_save_dir, exist_ok=True)
        os.makedirs(df_save_dir, exist_ok=True)
        save_filename = f"[{save_prefix}]_O{stats.net_stat.o_ratio:.3f}_S{stats.net_stat.sharpe_ratio:.3f}"
        plot_save_fp = f"{plot_save_dir}/{save_filename}" + ".png"
        backtest_dataframe_fp = f"{df_save_dir}/{save_filename}" + ".parquet"
        return plot_save_fp, backtest_dataframe_fp
