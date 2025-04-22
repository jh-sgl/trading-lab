from typing import Any

import lightning as L
import pandas as pd
import torch
import torch.nn as nn
import torchmetrics as tm
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LRScheduler

from util.registry import build_loss_func, build_network, register_model

from .const import Key, Num


@register_model("STReLUpstream")
class STReLUpstreamModel(L.LightningModule):
    def __init__(
        self,
        network: DictConfig,
        lr: float,
        regr_loss_func: DictConfig,
        clsf_loss_func: DictConfig,
        reco_loss_func: DictConfig,
        rank_loss_func: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self._network = build_network(network)
        self._regr_loss_func = build_loss_func(regr_loss_func)
        self._clsf_loss_func = build_loss_func(clsf_loss_func)
        self._reco_loss_func = build_loss_func(reco_loss_func)
        self._rank_loss_func = build_loss_func(rank_loss_func)
        self._lr = lr

        self._train_avg_past_loss = tm.MeanMetric()
        self._train_avg_future_loss = tm.MeanMetric()
        self._train_avg_total_loss = tm.MeanMetric()

        self._val_avg_past_loss = tm.MeanMetric()
        self._val_avg_future_loss = tm.MeanMetric()
        self._val_avg_total_loss = tm.MeanMetric()

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

    def _compute_loss(
        self, outputs: dict[Key, torch.Tensor], labels: dict[Key, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        def _normalize(labels: torch.Tensor) -> torch.Tensor:
            return (labels - labels.mean(dim=1, keepdim=True)) / (
                labels.std(dim=1, keepdim=True, unbiased=False) + Num.EPS
            )

        regr_loss = self._regr_loss_func(outputs[Key.REGRESSION], _normalize(labels[Key.REGRESSION]))
        clsf_loss = self._clsf_loss_func(outputs[Key.CLASSIFICATION], labels[Key.CLASSIFICATION])
        reco_loss = self._reco_loss_func(outputs[Key.RECONSTRUCTION], _normalize(labels[Key.RECONSTRUCTION]))
        rank_loss = self._rank_loss_func(outputs[Key.RANKING], labels[Key.RANKING])
        return regr_loss, clsf_loss, reco_loss, rank_loss

    def _step(self, batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        past_inputs, past_labels, future_labels = batch

        past_outputs, future_outputs = self._network(past_inputs)

        past_regr_loss, past_clsf_loss, past_reco_loss, past_rank_loss = self._compute_loss(past_outputs, past_labels)
        future_regr_loss, future_clsf_loss, future_reco_loss, future_rank_loss = self._compute_loss(
            future_outputs, future_labels
        )

        past_loss = past_regr_loss + past_clsf_loss + past_reco_loss + past_rank_loss
        future_loss = future_regr_loss + future_clsf_loss + future_reco_loss + future_rank_loss
        total_loss = past_loss + future_loss
        return past_loss, future_loss, total_loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        past_loss, future_loss, total_loss = self._step(batch)

        self._train_avg_past_loss(past_loss)
        self._train_avg_future_loss(future_loss)
        self._train_avg_total_loss(total_loss)
        self.log("train_past_loss", self._train_avg_past_loss, on_epoch=True)
        self.log("train_future_loss", self._train_avg_future_loss, on_epoch=True)
        self.log("train_total_loss", self._train_avg_total_loss, on_epoch=True, prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx) -> None:
        past_loss, future_loss, total_loss = self._step(batch)

        self._val_avg_past_loss(past_loss)
        self._val_avg_future_loss(future_loss)
        self._val_avg_total_loss(total_loss)
        self.log("val_past_loss", self._val_avg_past_loss, on_epoch=True)
        self.log("val_future_loss", self._val_avg_future_loss, on_epoch=True)
        self.log("val_total_loss", self._val_avg_total_loss, on_epoch=True, prog_bar=True)


@register_model("STReLDownstream")
class STReLDownstreamModel(L.LightningModule):
    def __init__(
        self,
        upstream_model_fp: str,
        network: DictConfig,
        lr: float,
        loss_func: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        upstream_model = STReLUpstreamModel.load_from_checkpoint(upstream_model_fp)
        # upstream_model.train()
        # upstream_model.unfreeze()
        upstream_model.eval()
        upstream_model.freeze()

        self._network = build_network(network)
        self._network.init_network(upstream_model.network)

        self._loss_func = build_loss_func(loss_func)
        self._lr = lr

        self._train_loss = tm.MeanMetric()
        self._val_loss = tm.MeanMetric()

    def configure_optimizers(self) -> dict[str, Any]:
        optim = torch.optim.AdamW(self._network.parameters(), lr=self._lr, weight_decay=0.01)
        # lr_scheduler = CosineAnnealingWarmRestarts(optim, T_0=10, T_mult=2)
        return {
            "optimizer": optim,
            # "lr_scheduler": {"scheduler": lr_scheduler, "interval": "epoch", "frequency": 1},
        }

    def _compute_loss(self, outputs: dict[Key, torch.Tensor], labels: dict[Key, torch.Tensor]) -> torch.Tensor:
        return self._loss_func(outputs, labels)

    def _step(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        inputs, labels, _ = batch
        outputs = self._network(inputs)
        loss = self._loss_func(outputs, labels)
        return outputs, loss

    def _should_validate(self) -> bool:
        return (self.trainer.current_epoch + 1) % self.trainer.check_val_every_n_epoch == 0

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        _, loss = self._step(batch)
        self._train_loss(loss)
        self.log("train_loss", self._train_loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        outputs, loss = self._step(batch)
        self._val_loss(loss)
        self.log("val_loss", self._val_loss, on_epoch=True, prog_bar=True)

        timestamp = batch[-1]
        self._store_outputs_to_df(self.trainer.datamodule.val_df, outputs, timestamp)

    def _store_outputs_to_df(self, df: pd.DataFrame, outputs: torch.Tensor, timestamp: torch.Tensor) -> None:
        outputs = outputs.detach().cpu().numpy()
        timestamp = timestamp.detach().cpu().numpy()

        output_df = pd.DataFrame(
            {
                Key.OUTPUT_PRED: outputs,
                Key.TIMESTAMP: timestamp,
            }
        )
        output_df[Key.TIMESTAMP] = output_df[Key.TIMESTAMP].apply(lambda x: pd.to_datetime(x))
        output_df = output_df.set_index(Key.TIMESTAMP)
        output_df[Key.TIMESTAMP] = output_df.index

        df.loc[output_df.index, Key.OUTPUT_PRED] = output_df[Key.OUTPUT_PRED]
