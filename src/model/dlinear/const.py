from enum import Enum

import torch

CacheDict = dict[str, list[torch.Tensor] | torch.Tensor]


class CacheKey(str, Enum):
    OUTPUT = "output"
    LABEL_TS = "label_ts"
    LABEL_PRICE_OPEN = "label_price_open"
    LABEL_PRICE_CLOSE = "label_price_close"
    DECISION = "decision"
