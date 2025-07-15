import logging
from typing import Callable

import numpy as np
import pandas as pd
from numba import jit, njit
from omegaconf import DictConfig

from ...util.const import DFKey, Num
from ...util.registry import (
    build_stop_loss_func,
    build_take_profit_func,
    register_backtester,
    register_stop_loss_func,
    register_take_profit_func,
)


@register_stop_loss_func("no_stop")
def make_no_stop_loss_func():
    @njit
    def no_stop_loss_func(price_path, entry_price, signal):
        return -1

    return no_stop_loss_func


@register_take_profit_func("no_take")
def make_no_take_profit_func():
    @njit
    def no_take_profit_func(price_path, entry_price, signal):
        return -1

    return no_take_profit_func


# @jit(forceobj=True)
@njit
def compute_trades_numba(
    prices,
    signals,
    signal_indices,
    eod_indices,
    commission_rate,
    slippage,
    stop_loss_func,
    take_profit_func,
):

    n_signals = len(signal_indices)

    entry_prices = np.empty(n_signals)
    exit_prices = np.empty(n_signals)
    entry_signals = np.zeros(n_signals, dtype=np.int64)
    raw_pnls = np.empty(n_signals)
    net_pnls = np.empty(n_signals)
    costs = np.empty(n_signals)
    exit_indices = np.empty(n_signals, dtype=np.int64)

    for i in range(n_signals):
        start = signal_indices[i]
        end = eod_indices[i]

        entry_price = prices[start]
        signal = signals[start]

        path_len = end - start + 1
        price_path = prices[start : end + 1]

        # Apply user-defined conditions
        sl_idx = stop_loss_func(price_path, entry_price, signal)
        tp_idx = take_profit_func(price_path, entry_price, signal)

        exit_rel_idx = path_len - 1  # EOD fallback
        if 0 <= sl_idx < path_len:
            exit_rel_idx = sl_idx
        if 0 <= tp_idx < path_len and (sl_idx == -1 or tp_idx < sl_idx):
            exit_rel_idx = tp_idx

        exit_idx = start + exit_rel_idx
        exit_price = prices[exit_idx]

        size = abs(signal)
        direction = 1 if signal > 0 else -1

        if exit_rel_idx == path_len - 1:
            slippage_coef = 1
        else:
            slippage_coef = 2

        raw_pnl = direction * size * (exit_price - entry_price)
        cost = size * (
            (entry_price + exit_price) * commission_rate + slippage_coef * slippage
        )
        net_pnl = raw_pnl - cost

        entry_prices[i] = entry_price
        exit_prices[i] = exit_price
        entry_signals[i] = signal
        raw_pnls[i] = raw_pnl
        net_pnls[i] = net_pnl
        costs[i] = cost
        exit_indices[i] = exit_idx

    return (
        entry_prices,
        exit_prices,
        entry_signals,
        raw_pnls,
        net_pnls,
        costs,
        exit_indices,
    )


@register_backtester("basic")
class Backtester:
    def __init__(
        self,
        stop_loss_func: DictConfig,
        take_profit_func: DictConfig,
        initial_balance: int,
        risk_exposure: float,
        volatility_coef: float,
        volatility_window: int,
        volatility_clip_range: tuple[float, float],
        signal_threshold: float,
        pred_to_signal_mode: str,
        position_size_strategy: str,
        signal_stop_trade_after_n_min: int | None,
        signal_trade_between_hours: tuple[str, str] | None,
        signal_dayofweek: int | None,
    ) -> None:
        self._stop_loss_func = build_stop_loss_func(stop_loss_func)
        self._take_profit_func = build_take_profit_func(take_profit_func)

        self._initial_balance = initial_balance
        self._risk_exposure = risk_exposure
        self._volatility_coef = volatility_coef
        self._volatility_window = volatility_window
        self._volatility_clip_range = volatility_clip_range

        self._signal_threshold = signal_threshold
        self._pred_to_signal_mode = pred_to_signal_mode
        self._position_size_strategy = position_size_strategy
        self._signal_stop_trade_after_n_min = signal_stop_trade_after_n_min
        self._signal_trade_between_hours = signal_trade_between_hours
        self._signal_dayofweek = signal_dayofweek

    def _calc_max_position_size(self, df: pd.DataFrame) -> pd.Series:
        volatility = (
            df[DFKey.PRICE_EXECUTION]
            .diff()
            .abs()
            .rolling(self._volatility_window, min_periods=1)
            .mean()
        )
        volatility = volatility.clip(*self._volatility_clip_range)
        return (self._initial_balance * self._risk_exposure) / (
            volatility * self._volatility_coef * Num.PRICE_MULTIPLIER + Num.EPS
        ).dropna()

    def _pred_to_signal(self, df: pd.DataFrame) -> pd.Series:
        if self._pred_to_signal_mode == "shl_wsum":
            output_pred = df[
                [
                    DFKey.OUTPUT_PRED_SHORT,
                    DFKey.OUTPUT_PRED_HOLD,
                    DFKey.OUTPUT_PRED_LONG,
                ]
            ].dropna()
            signal = output_pred.dot([-1, 0, 1])
        else:
            raise ValueError(
                f"Invalid pred_to_signal_mode: {self._pred_to_signal_mode}"
            )

        signal[abs(signal) < self._signal_threshold] = 0

        if self._position_size_strategy == "voladj":
            signal = signal * df[DFKey.MAX_POSITION_SIZE]
        elif self._position_size_strategy == "fixpos1":
            signal[signal > 0] = 1
            signal[signal < 0] = -1
            signal[abs(signal) != 1] = 0
        else:
            raise ValueError(
                f"Invalid position_size_strategy: {self._position_size_strategy}"
            )

        signal = np.round(signal, decimals=0)

        mask = pd.Series(False, index=df.index)

        if self._signal_stop_trade_after_n_min is not None:
            first_times = df.groupby(df.index.date).apply(lambda g: g.index.min())
            for _, first_time in first_times.items():
                start = first_time
                end = start + pd.Timedelta(minutes=self._signal_stop_trade_after_n_min)
                mask |= (df.index >= start) & (df.index <= end)

        if self._signal_trade_between_hours is not None:
            start_hour, end_hour = self._signal_trade_between_hours
            mask |= (df.index.hour >= start_hour) & (df.index.hour <= end_hour)

        if self._signal_dayofweek is not None:
            mask |= df.index.dayofweek == self._signal_dayofweek

        signal[~mask] = 0

        return pd.Series(signal, index=output_pred.index)

    def _should_stop_loss(
        self, df: pd.DataFrame, direction: int, entry_price: float, cur_price: float
    ) -> bool:
        raise NotImplementedError("Subclass should implement this function")

    def _should_take_profit(
        self,
        df: pd.DataFrame,
        direction: int,
        entry_price: float,
        cur_price: float,
        entry_time: pd.DatetimeIndex,
        cur_time: pd.DatetimeIndex,
    ) -> bool:
        raise NotImplementedError("Subclass should implement this function")

    def _should_force_liquidation(
        self, df: pd.DataFrame, cur_time: pd.DatetimeIndex
    ) -> bool:
        return bool(df.at[cur_time, DFKey.DAILY_LAST_ROW])

    def backtest_with_numba(self, df: pd.DataFrame) -> pd.DataFrame | None:
        df[DFKey.MAX_POSITION_SIZE] = self._calc_max_position_size(df)
        df[DFKey.SIGNAL] = self._pred_to_signal(df)
        df[DFKey.SIGNAL] = df[DFKey.SIGNAL].fillna(0)

        if df[DFKey.SIGNAL].isna().all():
            logging.info("No trades found. Skip backtesting.")
            return None

        prices = df[DFKey.PRICE_EXECUTION].to_numpy()
        signals = df[DFKey.SIGNAL].to_numpy()
        dates = df[DFKey.DATE].to_numpy()
        timestamps = df.index.to_numpy()

        signal_mask = signals != 0
        signal_indices = np.where(signal_mask)[0]
        signal_dates = dates[signal_indices]

        # Build EOD map
        _, last_indices = np.unique(dates[::-1], return_index=True)
        last_indices = len(dates) - 1 - last_indices
        date_to_eod = dict(zip(dates[last_indices], last_indices))
        eod_indices = np.array([date_to_eod[d] for d in signal_dates])

        # Run Numba-accelerated simulation
        (
            entry_prices,
            exit_prices,
            entry_signals,
            raw_pnls,
            net_pnls,
            costs,
            exit_indices,
        ) = compute_trades_numba(
            prices,
            signals,
            signal_indices,
            eod_indices,
            Num.COMMISSION_RATE,
            Num.SLIPPAGE_PER_EXECUTION,
            self._stop_loss_func,
            self._take_profit_func,
        )

        # Construct results DataFrame
        results_df = pd.DataFrame(
            {
                DFKey.ENTRY_TIME: timestamps[signal_indices],
                DFKey.EXIT_TIME: timestamps[exit_indices],
                DFKey.ENTRY_PRICE: entry_prices,
                DFKey.EXIT_PRICE: exit_prices,
                DFKey.ENTRY_SIGNAL: entry_signals,
                DFKey.RAW_PNL: raw_pnls,
                DFKey.NET_PNL: net_pnls,
                DFKey.RAW_PNL_UWON: raw_pnls * Num.PRICE_MULTIPLIER / Num.UWON,
                DFKey.NET_PNL_UWON: net_pnls * Num.PRICE_MULTIPLIER / Num.UWON,
                DFKey.COST: costs,
            }
        ).set_index(DFKey.ENTRY_TIME)

        for col in results_df.columns:
            df[col] = results_df[col]

        df[
            [
                DFKey.RAW_PNL,
                DFKey.NET_PNL,
                DFKey.RAW_PNL_UWON,
                DFKey.NET_PNL_UWON,
                DFKey.COST,
                DFKey.ENTRY_SIGNAL,
            ]
        ] = df[
            [
                DFKey.RAW_PNL,
                DFKey.NET_PNL,
                DFKey.RAW_PNL_UWON,
                DFKey.NET_PNL_UWON,
                DFKey.COST,
                DFKey.ENTRY_SIGNAL,
            ]
        ].fillna(
            0
        )

        return df

    def backtest_vectorized(
        self,
        df: pd.DataFrame,
        stop_loss_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        take_profit_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> pd.DataFrame | None:
        """
        Backtest the generated signals using stop-loss and take-profit criteria.
        Accumulate positions if repeated signals occur.
        """
        # TODO: implement position sizing

        df[DFKey.PRICE_EXECUTION] = (
            df.groupby(df.index.date)[[DFKey.FUTURE_PRICE_OPEN]].shift(-1).ffill()
        )

        timestamps = df.index.to_numpy()
        prices = df[DFKey.PRICE_EXECUTION].to_numpy()
        dates = df[DFKey.DATE].to_numpy()

        signals = df[DFKey.SIGNAL].to_numpy()
        signal_mask = signals != 0
        signal_indices = np.where(signal_mask)[0]

        entry_prices = prices[signal_indices]
        entry_signals = signals[signal_indices]
        entry_dates = dates[signal_indices]

        _, last_indices = np.unique(dates[::-1], return_index=True)
        last_indices = len(dates) - 1 - last_indices
        date_to_eod_idx = dict(zip(dates[last_indices], last_indices))
        eod_indices = np.array([date_to_eod_idx[d] for d in entry_dates])

        max_len = np.max(eod_indices - signal_indices) + 1
        n_signals = len(signal_indices)

        price_matrix = np.full((n_signals, max_len), np.nan)
        for i, (start, end) in enumerate(zip(signal_indices, eod_indices)):
            length = end - start + 1
            price_matrix[i, :length] = prices[start : end + 1]

        stop_mask = stop_loss_func(price_matrix, entry_prices)
        take_profit_mask = take_profit_func(price_matrix, entry_prices)

        stop_first_idx = np.argmax(stop_mask, axis=1)
        take_profit_first_idx = np.argmax(take_profit_mask, axis=1)

        stop_hit = stop_mask[np.arange(n_signals), stop_first_idx]
        take_profit_hit = take_profit_mask[np.arange(n_signals), take_profit_first_idx]

        eod_idx = np.full(n_signals, max_len - 1)

        exit_idx = np.where(stop_hit, stop_first_idx, eod_idx)
        take_profit_better_than_stop = take_profit_hit & (
            ~stop_hit | (take_profit_first_idx < stop_first_idx)
        )
        exit_idx = np.where(
            take_profit_better_than_stop, take_profit_first_idx, exit_idx
        )

        exit_indices = signal_indices + exit_idx
        exit_prices = prices[exit_indices]

        raw_pnl = np.where(
            entry_signals == 1, exit_prices - entry_prices, entry_prices - exit_prices
        )
        cost = (
            entry_prices + exit_prices
        ) * Num.COMMISSION_RATE + Num.SLIPPAGE_PER_EXECUTION * 2
        net_pnl = raw_pnl - cost

        trade_df = pd.DataFrame(
            {
                DFKey.ENTRY_TIME: timestamps[signal_indices],
                DFKey.EXIT_TIME: timestamps[exit_indices],
                DFKey.ENTRY_PRICE: entry_prices,
                DFKey.EXIT_PRICE: exit_prices,
                DFKey.ENTRY_SIGNAL: entry_signals,
                DFKey.RAW_PNL: raw_pnl,
                DFKey.NET_PNL: net_pnl,
                DFKey.RAW_PNL_UWON: raw_pnl * Num.PRICE_MULTIPLIER / Num.UWON,
                DFKey.NET_PNL_UWON: net_pnl * Num.PRICE_MULTIPLIER / Num.UWON,
                DFKey.COST: cost,
            }
        ).set_index(DFKey.ENTRY_TIME)

        for col in trade_df.columns:
            df[col] = trade_df[col]

        cols_to_fillna = [
            DFKey.RAW_PNL,
            DFKey.NET_PNL,
            DFKey.RAW_PNL_UWON,
            DFKey.NET_PNL_UWON,
            DFKey.ENTRY_SIGNAL,
            DFKey.COST,
        ]
        df[cols_to_fillna] = df[cols_to_fillna].fillna(0)
        return df

    def _should_stop_loss_vectorized(
        self, df: pd.DataFrame, entry_signal: int, entry_price: float
    ) -> pd.Timestamp | None:
        raise NotImplementedError("Subclass should implement this function")

    def _should_take_profit_vectorized(
        self,
        df: pd.DataFrame,
        entry_signal: int,
        entry_price: float,
        entry_time: pd.Timestamp,
    ) -> pd.Timestamp | None:
        raise NotImplementedError("Subclass should implement this function")

    def backtest(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """
        Backtest the generated signals using stop-loss and take-profit criteria.
        Accumulate positions if repeated signals occur.
        """
        # TODO: implement position sizing

        df[DFKey.PRICE_EXECUTION] = (
            df.groupby(df.index.date)[[DFKey.FUTURE_PRICE_OPEN]].shift(-1).ffill()
        )
        df[DFKey.DAILY_LAST_ROW] = False
        df.loc[df.groupby(df.index.date).tail(1).index, DFKey.DAILY_LAST_ROW] = True

        trades = []
        open_positions = []
        for row in df[[DFKey.PRICE_EXECUTION, DFKey.SIGNAL]].itertuples(index=True):
            cur_time = row.Index
            cur_price = row[1]
            cur_signal = row[2]

            # Check if any open position should be closed
            remaining_positions = []
            for entry in open_positions:
                entry_signal, entry_time, entry_price = entry
                if entry_signal == 1:  # long
                    raw_pnl = cur_price - entry_price
                elif entry_signal == -1:  # short
                    raw_pnl = entry_price - cur_price
                cost = (
                    entry_price + cur_price
                ) * Num.COMMISSION_RATE + Num.SLIPPAGE_PER_EXECUTION * 2
                net_pnl = raw_pnl - cost

                if (
                    self._should_stop_loss(df, entry_signal, entry_price, cur_price)
                    or self._should_take_profit(
                        df,
                        entry_signal,
                        entry_price,
                        cur_price,
                        entry_time,
                        cur_time,
                    )
                    or self._should_force_liquidation(df, cur_time)
                ):
                    trades.append(
                        {
                            DFKey.ENTRY_TIME: entry_time,
                            DFKey.EXIT_TIME: cur_time,
                            DFKey.ENTRY_PRICE: entry_price,
                            DFKey.EXIT_PRICE: cur_price,
                            DFKey.ENTRY_SIGNAL: entry_signal,
                            DFKey.RAW_PNL: raw_pnl,
                            DFKey.NET_PNL: net_pnl,
                            DFKey.RAW_PNL_UWON: raw_pnl
                            * Num.PRICE_MULTIPLIER
                            / Num.UWON,
                            DFKey.NET_PNL_UWON: net_pnl
                            * Num.PRICE_MULTIPLIER
                            / Num.UWON,
                            DFKey.COST: cost,
                        }
                    )
                else:
                    remaining_positions.append(entry)

            open_positions = remaining_positions

            # Accumulate new position
            if cur_signal != 0:
                open_positions.append((cur_signal, cur_time, cur_price))

        trade_df = pd.DataFrame(trades).set_index(DFKey.ENTRY_TIME)

        for col in trade_df.columns:
            df[col] = trade_df[col]

        cols_to_fillna = [
            DFKey.RAW_PNL,
            DFKey.NET_PNL,
            DFKey.RAW_PNL_UWON,
            DFKey.NET_PNL_UWON,
            DFKey.ENTRY_SIGNAL,
            DFKey.COST,
        ]
        df[cols_to_fillna] = df[cols_to_fillna].fillna(0)
        return df
