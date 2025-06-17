from enum import Enum


class DFKey(str, Enum):
    FUTURE_PRICE_OPEN = "future_price_open"
    FUTURE_PRICE_HIGH = "future_price_high"
    FUTURE_PRICE_LOW = "future_price_low"
    FUTURE_PRICE_CLOSE = "future_price_close"
    FUTURE_VOLUME = "future_volume"

    TIMESTAMP = "timestamp"

    PRICE_EXECUTION = "price_execution"
    LONG_PROFIT_TRAJECTORY = "long_profit_trajectory"
    SHORT_PROFIT_TRAJECTORY = "short_profit_trajectory"
    LONG_PROFIT_TRAJECTORY_COSTLESS = "long_profit_trajectory_costless"
    SHORT_PROFIT_TRAJECTORY_COSTLESS = "short_profit_trajectory_costless"
    HOLD_PROFIT_TRAJECTORY = "hold_profit_trajectory"

    VOLATILITY_50 = "price_volatility_50"

    OUTPUT_PRED = "output_pred"

    PROFIT = "profit"
    PROFIT_COSTLESS = "profit_costless"
    PROFIT_UWON = "profit_uwon"
    PROFIT_UWON_COSTLESS = "profit_uwon_costless"
    POSITION = "position"
