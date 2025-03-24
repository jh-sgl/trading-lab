from typing import Any, Literal

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics as tm
from torch.optim import Optimizer

from jh.model.common.loss import FocalLoss
from jh.model.dlinear.network import DLinear


class DLinearModel(L.LightningModule):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        input_ch: int,
        moving_avg_kernel_size: int,
        lr: float,
        loss_func: Literal["focal_loss", "l1_loss"],
        loss_func_args: dict[str, Any] = {},
    ) -> None:
        super().__init__()
        self._network = DLinear(seq_len, pred_len, input_ch, moving_avg_kernel_size)
        self._lr = lr
        self._loss_func = self._select_loss_func(loss_func, loss_func_args)

        self._train_cache: dict[str, list[torch.Tensor] | torch.Tensor] = {
            "output": [],
            "output_sorted": [],
            "label_timestamp": [],
            "label_timestamp_sorted": [],
            "label_price_open": [],
            "label_price_open_sorted": [],
            "label_price_close": [],
            "label_price_close_sorted": [],
        }
        self._val_cache = self._train_cache.copy()
        self._val_avg_loss = tm.MeanMetric()

    def _select_loss_func(
        self, loss_func: Literal["focal_loss", "l1_loss"], loss_func_args: dict[str, Any]
    ) -> nn.Module:
        if loss_func == "focal_loss":
            return FocalLoss(**loss_func_args)
        elif loss_func == "l1_loss":
            return nn.L1Loss(**loss_func_args)

    def _use_train_cache(self) -> bool:
        return self.trainer.current_epoch % self.trainer.check_val_every_n_epoch == 0

    @property
    def train_cache(self) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        return self._train_cache

    @property
    def val_cache(self) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        return self._val_cache

    def _calc_loss(self, output: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return self._loss_func(output, label)

    def _prepare_input(self, input_batch: torch.Tensor) -> torch.Tensor:
        return input_batch

    def _prepare_label(self, label_batch: torch.Tensor) -> torch.Tensor:
        return label_batch

    def configure_optimizers(self) -> Optimizer:
        optim = torch.optim.Adam(self._network.parameters(), lr=self._lr)
        return optim

    def _step(self, batch, cache: dict[str, list[Any]] | None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        (
            input_batch,
            label_batch,
            _,
            label_timestamp,
            label_price_open,
            label_price_close,
        ) = batch

        input_ = self._prepare_input(input_batch)
        label = self._prepare_label(label_batch)

        output = self._network(input_)
        output = F.softmax(output, dim=1)
        loss = self._calc_loss(output, label)

        if cache is not None:
            cache["output"].append(output)
            cache["label_timestamp"].append(label_timestamp)
            cache["label_price_open"].append(label_price_open)
            cache["label_price_close"].append(label_price_close)

        return loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        cache = self._train_cache if self._use_train_cache else None
        loss = self._step(batch, cache)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        loss = self._step(batch, cache=self._val_cache)
        self._val_avg_loss(loss)
        self.log("val_loss", self._val_avg_loss, on_step=True, on_epoch=True)

    def on_train_epoch_start(self) -> None:
        for key in self._train_cache:
            self._train_cache[key] = []

    def on_validation_epoch_start(self) -> None:
        for key in self._val_cache:
            self._val_cache[key] = []

    def on_validation_epoch_end(self) -> None:
        def _process_cache(
            cache: dict[str, list[torch.Tensor] | torch.Tensor],
            output_tensor_key: str = "output",
            sort_tensor_key: str = "label_timestamp",
        ) -> None:
            for k, v in cache.items():
                if isinstance(v, list):
                    cache[k] = torch.concat(cache[k])

            cache["decision"] = self._output_to_decision(cache[output_tensor_key])

            sorted_tensor_key = sort_tensor_key + "_sorted"
            cache[sorted_tensor_key], sort_idx = cache[sort_tensor_key].flatten().sort()
            for key in cache.keys() - {sorted_tensor_key}:
                cache[key + "_sorted"] = cache[key].flatten()[sort_idx]

        if self.trainer.current_epoch > 0:
            _process_cache(self._train_cache)
            _process_cache(self._val_cache)

    def _output_to_decision(self, output: torch.Tensor) -> torch.Tensor:
        decision = output[:, 0] * -1 + output[:, -1]
        return decision
