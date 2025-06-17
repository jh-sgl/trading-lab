from enum import Enum

import torch

CacheDict = dict[str, list[torch.Tensor] | torch.Tensor]


class CacheKey(str, Enum):
    TS_LABEL = "ts_label"
    BAND_OFFSET = "band_offset"
    BAND_CENTER_PRED = "band_center_pred"
    TODAY_CUTOFF_MEAN_PRICE = "today_cutoff_mean_price"
