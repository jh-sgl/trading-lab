from enum import Enum


class DFKey(str, Enum):
    FUTURE_PRICE_OPEN = "future_price_open"
    FUTURE_PRICE_HIGH = "future_price_high"
    FUTURE_PRICE_LOW = "future_price_low"
    FUTURE_PRICE_CLOSE = "future_price_close"
    FUTURE_VOLUME = "future_volume"
    FOREIGN_TRADE_CLOSE = "foreign_trade_close"

    TIMESTAMP = "timestamp"

    PRICE_EXECUTION = "price_execution"
    LONG_PROFIT_TRAJECTORY = "long_profit_trajectory"
    SHORT_PROFIT_TRAJECTORY = "short_profit_trajectory"
    LONG_PROFIT_TRAJECTORY_COSTLESS = "long_profit_trajectory_costless"
    SHORT_PROFIT_TRAJECTORY_COSTLESS = "short_profit_trajectory_costless"
    HOLD_PROFIT_TRAJECTORY = "hold_profit_trajectory"

    VOLATILITY_50 = "price_volatility_50"

    OUTPUT_PRED = "output_pred"

    DAILY_LAST_ROW = "daily_last_row"

    ORIG_SIGNAL = "orig_signal"
    ENTRY_TIME = "entry_time"
    EXIT_TIME = "cur_time"
    ENTRY_SIGNAL = "entry_signal"
    ENTRY_PRICE = "entry_price"
    EXIT_PRICE = "cur_price"
    RAW_PNL = "raw_pnl"
    NET_PNL = "net_pnl"
    RAW_PNL_UWON = "raw_pnl_uwon"
    NET_PNL_UWON = "net_pnl_uwon"
    COST = "cost"

    MODEL_SIGNAL = "model_signal"
