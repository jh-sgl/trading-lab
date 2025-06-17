from enum import Enum

import torch

CacheDict = dict[str, list[torch.Tensor] | torch.Tensor]


class Key(str, Enum):
    FUTURE_PRICE_OPEN = "future_price_open"
    FUTURE_PRICE_HIGH = "future_price_high"
    FUTURE_PRICE_LOW = "future_price_low"
    FUTURE_PRICE_CLOSE = "future_price_close"
    FUTURE_VOLUME = "future_volume"

    TIMESTAMP = "timestamp"

    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    RECONSTRUCTION = "reconstruction"
    RANKING = "ranking"

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


class Num:
    COMMISSION_RATE = 0.00029
    SLIPPAGE_PER_EXECUTION = 0.05
    EPS = 1e-25
    UWON = 100_000_000
