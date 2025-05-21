from typing import Any

import lightning as L
import pandas as pd
import torch
import torch.nn as nn
import torchmetrics as tm
from omegaconf import DictConfig
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from util.registry import build_loss_func, build_network, register_model

from .const import DFKey


@register_model("ExtremaDetectorRevis")
class ExtremaDetectorRevisModel(L.LightningModule):
    def __init__(self, network: DictConfig, loss_func: DictConfig, lr: float, use_soft_label: bool) -> None:
        super().__init__()
        self.save_hyperparameters()

        self._network = build_network(network)
        self._loss_func = build_loss_func(loss_func)
        self._lr = lr
        self._use_soft_label = use_soft_label

        self._train_avg_loss = tm.MeanMetric()
        self._val_avg_loss = tm.MeanMetric()

    @property
    def network(self) -> nn.Module:
        return self._network

    def configure_optimizers(self) -> dict[str, Any]:
        optim = torch.optim.AdamW(self._network.parameters(), lr=self._lr)
        lr_scheduler = CosineAnnealingWarmRestarts(optim, T_0=10, T_mult=2)
        return {
            "optimizer": optim,
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": "epoch", "frequency": 1},
        }

    def _compute_loss(self, outputs: dict[DFKey, torch.Tensor], labels: dict[DFKey, torch.Tensor]) -> torch.Tensor:
        loss = self._loss_func(outputs, labels)
        return loss

    def _step(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        inputs, labels, _ = batch

        outputs = self._network(inputs)
        loss = self._compute_loss(outputs, labels)
        return outputs, loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        outputs, loss = self._step(batch)
        self._train_avg_loss(loss)
        self.log("train_loss", self._train_avg_loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        outputs, loss = self._step(batch)
        self._val_avg_loss(loss)
        self.log("val_loss", self._val_avg_loss, on_epoch=True, prog_bar=True)
        timestamp = batch[-1]
        self._store_outputs_to_df(self.trainer.datamodule.val_df, outputs, timestamp)

    def _store_outputs_to_df(self, df: pd.DataFrame, outputs: torch.Tensor, timestamp: torch.Tensor) -> None:
        outputs = outputs.detach().cpu().numpy()
        timestamp = timestamp.detach().cpu().numpy()

        output_df = pd.DataFrame(
            {
                DFKey.OUTPUT_PRED: outputs,
                DFKey.TIMESTAMP: timestamp,
            }
        )
        output_df[DFKey.TIMESTAMP] = output_df[DFKey.TIMESTAMP].apply(lambda x: pd.to_datetime(x))
        output_df = output_df.set_index(DFKey.TIMESTAMP)
        output_df[DFKey.TIMESTAMP] = output_df.index

        df.loc[output_df.index, DFKey.OUTPUT_PRED] = output_df[DFKey.OUTPUT_PRED]
