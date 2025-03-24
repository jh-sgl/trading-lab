from abc import abstractmethod
from enum import Enum

import lightning as L
import pandas as pd
import torch


class Decision(Enum):
    BUY = 1
    HOLD = 0
    SELL = -1


class ModelBase(ABC, L.LightningModule):
    def __init__(self) -> None:
        super.__init__()

    @abstractmethod
    def _output_to_decision(self, output: torch.Tensor) -> pd.DataFrame:
        pass
