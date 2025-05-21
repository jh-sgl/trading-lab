import numpy as np
import pandas as pd
from omegaconf import DictConfig

from util.const import Num

from .const import DFKey

STRATEGY_REGISTRY = {}


class Strategy:
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df.dropna(subset=self.get_columns_to_use()).reset_index().set_index("time")
        self._df[DFKey.PRICE_EXECUTION] = (
            self._df.groupby(self._df.index.date)[DFKey.FUTURE_PRICE_OPEN].shift(-1).ffill()
        )

        self._df[DFKey.DAILY_LAST_ROW] = False
        self._df.loc[self._df.groupby(self._df.index.date).tail(1).index, DFKey.DAILY_LAST_ROW] = True

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def _should_stop_loss(self, direction: int, entry_price: float, cur_price: float) -> bool:
        raise NotImplementedError("Subclass should implement this function")

    def _should_take_profit(
        self, direction: int, entry_price: float, cur_price: float, cur_time: int, entry_time: int, df: pd.DataFrame
    ) -> bool:
        raise NotImplementedError("Subclass should implement this function")

    def _should_force_liquidation(self, cur_time: pd.DatetimeIndex):
        return self._df.loc[cur_time, DFKey.DAILY_LAST_ROW]

    def _generate_signals(self):
        """
        Generate signals using a user-defined logic function.
        The function should take in the price series and return a Series of 1, -1, or 0.
        """
        raise NotImplementedError("Subclass should implement this function (sell = -1 / hold = 0 / buy = 1)")

    def backtest(self, signal_type: str = "strategy") -> pd.DataFrame | None:
        """
        Backtest the generated signals using stop-loss and take-profit criteria.
        Accumulate positions if repeated signals occur.
        """

        if signal_type == "model":
            signal_key = DFKey.MODEL_SIGNAL
        elif signal_type == "strategy":
            self._generate_signals()
            signal_key = DFKey.ORIG_SIGNAL
        else:
            raise ValueError(f"Unknown signal type: {signal_type}")

        trades = []
        open_positions = []
        for row in self._df[[DFKey.PRICE_EXECUTION, signal_key]].itertuples(index=True):
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
                cost = (entry_price + cur_price) * Num.COMMISSION_RATE + Num.SLIPPAGE_PER_EXECUTION * 2
                net_pnl = raw_pnl - cost

                if (
                    self._should_stop_loss(entry_signal, entry_price, cur_price)
                    or self._should_take_profit(entry_signal, entry_price, cur_price, entry_time, cur_time, self._df)
                    or self._should_force_liquidation(cur_time)
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
                            DFKey.RAW_PNL_UWON: raw_pnl * Num.PRICE_MULTIPLIER / Num.UWON,
                            DFKey.NET_PNL_UWON: net_pnl * Num.PRICE_MULTIPLIER / Num.UWON,
                            DFKey.COST: cost,
                        }
                    )
                else:
                    remaining_positions.append(entry)

            open_positions = remaining_positions

            # Accumulate new position
            if cur_signal != 0:
                open_positions.append((cur_signal, cur_time, cur_price))

        if len(trades) == 0:
            return None

        trade_df = pd.DataFrame(trades).set_index(DFKey.ENTRY_TIME)

        df_with_result = self._df.copy()

        for col in trade_df.columns:
            df_with_result[col] = trade_df[col]

        cols_to_fillna = [
            DFKey.RAW_PNL,
            DFKey.NET_PNL,
            DFKey.RAW_PNL_UWON,
            DFKey.NET_PNL_UWON,
            DFKey.ENTRY_SIGNAL,
            DFKey.COST,
        ]
        df_with_result[cols_to_fillna] = df_with_result[cols_to_fillna].fillna(0)
        return df_with_result

    def update_df(self, df: pd.DataFrame) -> pd.DataFrame:
        self._df = df
        return self._df

    def get_columns_info_for_model(self) -> tuple[dict[str, str | None], str]:
        raise NotImplementedError("Subclass should specify column names for model inputs")

    def update_model_signal(self, model_signal: pd.Series | np.ndarray) -> None:
        self._df[DFKey.MODEL_SIGNAL] = model_signal

    def get_columns_to_use(self) -> list[str]:
        raise NotImplementedError("Subclass should specify columns to use for trading conditions")


def build_strategy(strategy_cfg: DictConfig, dataset_df: pd.DataFrame) -> Strategy:
    return STRATEGY_REGISTRY[strategy_cfg.name.lower()](dataset_df, **strategy_cfg.args)


def register_strategy(name):
    def wrapper(cls):
        if name in STRATEGY_REGISTRY:
            raise ValueError(f"Duplicate strategy registry key: {name}")
        STRATEGY_REGISTRY[name.lower()] = cls
        return cls

    return wrapper


@register_strategy("foreignvolumeprice")
class ForeignVolumePriceStrategy(Strategy):
    def __init__(self, df: pd.DataFrame, hold_time_in_min: int = 5):
        super().__init__(df)
        self._hold_time_in_min = hold_time_in_min

    def _should_stop_loss(self, direction: int, entry_price: float, cur_price: float):
        return cur_price < entry_price * 0.9

    def _should_take_profit(
        self, direction: int, entry_price: float, cur_price: float, cur_time: int, entry_time: int, df: pd.DataFrame
    ):
        return (cur_time - entry_time) >= pd.Timedelta(minutes=self._hold_time_in_min)

    def _generate_signals(self):
        df = self._df
        df["ma_price"] = df[DFKey.FUTURE_PRICE_CLOSE].rolling(60).mean()
        df["ma_volume"] = df[DFKey.FUTURE_VOLUME].rolling(10).mean()
        df["ma_foreign"] = df[DFKey.FOREIGN_TRADE_CLOSE].rolling(30).mean()

        df["vol_disparity"] = df[DFKey.FUTURE_VOLUME] - df["ma_volume"]
        df["price_disparity"] = df[DFKey.FUTURE_PRICE_CLOSE] - df["ma_price"]
        df["foreign_disparity"] = df[DFKey.FOREIGN_TRADE_CLOSE] - df["ma_foreign"]
        condition = (
            (df[DFKey.FOREIGN_TRADE_CLOSE] > 0)
            & (df["vol_disparity"] > 0)
            & (df["price_disparity"] > 0)
            & (df["foreign_disparity"] > 0)
        )
        df[DFKey.ORIG_SIGNAL] = condition.astype(int)  # 1 = buy, 0 = hold

    def get_columns_info_for_model(self) -> tuple[dict[str, str | None], str]:
        return {
            "ma_price": "standardized_normal",
            "ma_volume": "standardized_normal",
            "ma_foreign": "standardized_normal",
            DFKey.FOREIGN_TRADE_CLOSE: "standardized_normal",
            DFKey.FUTURE_VOLUME: "standardized_normal",
            DFKey.FUTURE_PRICE_CLOSE: "standardized_normal",
            "vol_disparity": "standardized_normal",
            "price_disparity": "standardized_normal",
            "foreign_disparity": "standardized_normal",
            DFKey.ORIG_SIGNAL: None,
        }, DFKey.NET_PNL

    def get_columns_to_use(self) -> list[str]:
        return [
            DFKey.FUTURE_PRICE_CLOSE,
            DFKey.FUTURE_VOLUME,
            DFKey.FOREIGN_TRADE_CLOSE,
        ]


@register_strategy("momentumforeignflow")
class MomentumForeignFlowStrategy(Strategy):
    def __init__(self, df: pd.DataFrame, window: int = 5, price_thresh: float = 0.005, hold_time_in_min: int = 5):
        super().__init__(df)
        self._window = window
        self._price_thresh = price_thresh
        self._hold_time_in_min = hold_time_in_min

    def _should_stop_loss(self, direction: int, entry_price: float, cur_price: float):
        if direction == 1:
            return cur_price < entry_price * 0.995
        else:
            return cur_price > entry_price * 1.005

    def _should_take_profit(
        self, direction: int, entry_price: float, cur_price: float, cur_time: int, entry_time: int, df: pd.DataFrame
    ):
        return (cur_time - entry_time) >= pd.Timedelta(minutes=self._hold_time_in_min)

    def _generate_signals(self):
        df = self._df

        # Momentum price change
        df["price_return"] = df[DFKey.FUTURE_PRICE_CLOSE].pct_change(self._window)

        # Volume spike relative to rolling average
        df["vol_avg"] = df[DFKey.FUTURE_VOLUME].rolling(self._window).mean()
        df["vol_sum"] = df[DFKey.FUTURE_VOLUME].rolling(self._window).sum()
        # df["vol_disparity"] = df["vol_sum"] - df["vol_avg"] * self._window

        # Foreign flow sum over window
        df["foreign_flow_sum"] = df[DFKey.FOREIGN_TRADE_CLOSE].rolling(self._window).sum()

        long_cond = (df["price_return"] > self._price_thresh) & (df["foreign_flow_sum"] > 0)
        # long_cond = (df["price_return"] > self._price_thresh) & (df["vol_disparity"] > 0) & (df["foreign_flow_sum"] > 0)

        short_cond = (df["price_return"] < -self._price_thresh) & (df["foreign_flow_sum"] < 0)
        # short_cond = (
        #     (df["price_return"] < -self._price_thresh) & (df["vol_disparity"] > 0) & (df["foreign_flow_sum"] < 0)
        # )

        df[DFKey.ORIG_SIGNAL] = 0
        df.loc[long_cond, DFKey.ORIG_SIGNAL] = 1
        df.loc[short_cond, DFKey.ORIG_SIGNAL] = -1

    def get_columns_info_for_model(self) -> tuple[dict[str, str | None], str]:
        return {
            "price_return": None,
            # "vol_disparity": "standardized_normal",
            # "foreign_flow_sum": "standardized_normal",
            DFKey.ORIG_SIGNAL: None,
        }, DFKey.NET_PNL

    def get_columns_to_use(self) -> list[str]:
        return [
            DFKey.FUTURE_PRICE_CLOSE,
            DFKey.FUTURE_VOLUME,
            DFKey.FOREIGN_TRADE_CLOSE,
        ]


@register_strategy("liquiditytrapreversal")
class LiquidityTrapReversalStrategy(Strategy):
    def __init__(
        self,
        df: pd.DataFrame,
        wick_window: int = 5,
        flow_window: int = 5,
        trap_thresh: float = 0.001,
    ):
        super().__init__(df)
        self._wick_window = wick_window
        self._flow_window = flow_window
        self._trap_thresh = trap_thresh

    def _should_stop_loss(self, direction: int, entry_price: float, cur_price: float):
        if direction == 1:
            return cur_price < entry_price * 0.997
        else:
            return cur_price > entry_price * 1.003

    def _should_take_profit(
        self, direction: int, entry_price: float, cur_price: float, cur_time: int, entry_time: int, df: pd.DataFrame
    ):
        # Exit after quick gain
        if direction == 1 and cur_price >= entry_price * 1.003:
            return True
        if direction == -1 and cur_price <= entry_price * 0.997:
            return True

        # Exit if trade stagnates
        cur_idx = df.index.get_loc(cur_time)
        entry_idx = df.index.get_loc(entry_time)

        if cur_idx - entry_idx >= 3:
            price_range = (
                df[DFKey.FUTURE_PRICE_CLOSE].iloc[entry_idx : cur_idx + 1].max()
                - df[DFKey.FUTURE_PRICE_CLOSE].iloc[entry_idx : cur_idx + 1].min()
            )
            if price_range < entry_price * 0.0008:
                return True

        return False

    def _generate_signals(self):
        df = self._df

        # Rolling high/low for trap detection
        df["rolling_low"] = df[DFKey.FUTURE_PRICE_LOW].rolling(self._wick_window, min_periods=1).min()
        df["rolling_high"] = df[DFKey.FUTURE_PRICE_HIGH].rolling(self._wick_window, min_periods=1).max()

        # Flow divergence: foreign vs institutional
        df["foreign_flow"] = df[DFKey.FOREIGN_TRADE_CLOSE].rolling(self._flow_window, min_periods=1).sum()
        df["inst_flow"] = df["institutional_trade_close"].rolling(self._flow_window, min_periods=1).sum()

        # Long setup: price makes new low but closes strong + foreign buying, institutional selling
        long_cond = (
            (df[DFKey.FUTURE_PRICE_LOW] < df["rolling_low"].shift(1) * (1 - self._trap_thresh))
            # & (df[DFKey.FUTURE_PRICE_CLOSE] > df[DFKey.FUTURE_PRICE_OPEN])
            & (df["foreign_flow"] > 0)
            & (df["inst_flow"] < 0)
        )

        # Short setup: price makes new high but closes weak + foreign selling, institutional buying
        short_cond = (
            (df[DFKey.FUTURE_PRICE_HIGH] > df["rolling_high"].shift(1) * (1 + self._trap_thresh))
            # & (df[DFKey.FUTURE_PRICE_CLOSE] < df[DFKey.FUTURE_PRICE_OPEN])
            & (df["foreign_flow"] < 0)
            & (df["inst_flow"] > 0)
        )

        df[DFKey.ORIG_SIGNAL] = 0
        df["long_cond"] = long_cond
        df["short_cond"] = short_cond
        df.loc[long_cond, DFKey.ORIG_SIGNAL] = 1
        df.loc[short_cond, DFKey.ORIG_SIGNAL] = -1

    def get_columns_info_for_model(self) -> tuple[dict[str, str | None], str]:
        return {
            "long_cond": None,
            "short_cond": None,
            "foreign_flow": None,
            "inst_flow": None,
            # DFKey.ORIG_SIGNAL: None,
        }, DFKey.NET_PNL

    def get_columns_to_use(self) -> list[str]:
        return [
            DFKey.FUTURE_PRICE_LOW,
            DFKey.FUTURE_PRICE_HIGH,
            DFKey.FUTURE_PRICE_CLOSE,
            DFKey.FUTURE_PRICE_OPEN,
            "institutional_trade_close",
            DFKey.FUTURE_VOLUME,
            DFKey.FOREIGN_TRADE_CLOSE,
        ]
