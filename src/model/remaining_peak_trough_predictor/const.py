from enum import Enum

import torch

CacheDict = dict[str, list[torch.Tensor] | torch.Tensor]


class DataKey(str, Enum):
    PRICE_ENTER = "price_enter"

    PRICE_EXIT_BUY = "price_exit_buy"
    PRICE_EXIT_SELL = "price_exit_sell"
    PRICE_MOVE_BUY = "price_move_buy"
    PRICE_MOVE_SELL = "price_move_sell"
    PRICE_MOVE_BUY_NORMALIZED = "price_move_buy_normalized"
    PRICE_MOVE_SELL_NORMALIZED = "price_move_sell_normalized"

    PRICE_ENTER_VOLATILITY_50 = "price_enter_volatility_50"

    OUTPUT_PRED_BUY = "output_pred_buy"
    OUTPUT_PRED_SELL = "output_pred_sell"
    OUTPUT_PRED_BUY_NORMALIZED = "output_pred_buy_normalized"
    OUTPUT_PRED_SELL_NORMALIZED = "output_pred_sell_normalized"

    TIMESTAMP = "timestamp"
