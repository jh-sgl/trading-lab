from enum import Enum

import torch

CacheDict = dict[str, list[torch.Tensor] | torch.Tensor]


class DataKey(str, Enum):
    PRICE_ENTER = "price_enter"
    PRICE_EXIT = "price_exit"
    PRICE_MOVE = "price_move"
    PRICE_ENTER_VOLATILITY_50 = "price_enter_volatility_50"
    OUTPUT_PRED = "output_pred"
    TIMESTAMP = "timestamp"
