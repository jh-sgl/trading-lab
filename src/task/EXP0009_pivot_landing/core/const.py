from enum import Enum


class DFKey(str, Enum):
    FUTURE_PRICE_OPEN = "future_price_open"
    FUTURE_PRICE_HIGH = "future_price_high"
    FUTURE_PRICE_LOW = "future_price_low"
    FUTURE_PRICE_CLOSE = "future_price_close"
    FUTURE_VOLUME = "future_volume"
    FUTURE_PRICE_MARKET_CLOSING = "future_price_market_closing"

    PRICE_EMA_5 = "price_ema_5"
    PRICE_EMA_20 = "price_ema_20"
    PRICE_EMA_60 = "price_ema_60"
    PRICE_EMA_120 = "price_ema_120"

    TIMESTAMP = "timestamp"

    PRICE_ENTER = "price_enter"
    PRICE_EXIT = "price_exit"
    PRICE_MOVE = "price_move"
    PRICE_MOVE_CLIPPED = "price_move_clipped"
    VOLATILITY_50 = "price_volatility_50"

    OUTPUT_PRED = "output_pred"

    PROFIT = "profit"
    PROFIT_UWON = "profit_uwon"
    POSITION = "position"
    COST = "cost"

    PIVOT = "pivot"
    R1 = "R1"
    S1 = "S1"
    R2 = "R2"
    S2 = "S2"
    R3 = "R3"
    S3 = "S3"
    MARKET_CLOSING_PIVOT_ZONE = "pivot_zone"
    PRICE_PIVOT_ZONE = "price_pivot_zone"
    PRICE_EMA_5_PIVOT_ZONE = "price_ema_5_pivot_zone"
    PRICE_EMA_20_PIVOT_ZONE = "price_ema_20_pivot_zone"
    PRICE_EMA_60_PIVOT_ZONE = "price_ema_60_pivot_zone"
    PRICE_EMA_120_PIVOT_ZONE = "price_ema_120_pivot_zone"
