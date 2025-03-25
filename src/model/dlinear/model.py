from enum import Enum
from typing import Any, Literal

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics as tm
from torch.optim import Optimizer

from model.common.loss import FocalLoss
from model.dlinear.network import DLinear

CacheDict = dict[str, list[torch.Tensor] | torch.Tensor]


class CacheKey(str, Enum):
    OUTPUT = "output"
    LABEL_TS = "label_ts"
    LABEL_PRICE_OPEN = "label_price_open"
    LABEL_PRICE_CLOSE = "label_price_close"
    DECISION = "decision"


class DLinearModel(L.LightningModule):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        input_ch: int,
        moving_avg_kernel_size: int,
        lr: float,
        loss_func: Literal[LossType.FOCAL, LossType.L1],
        loss_func_args: dict[str, Any] = {},
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self._network = DLinear(seq_len, pred_len, input_ch, moving_avg_kernel_size)
        self._lr = lr
        self._loss_func = self._build_loss_func(loss_func, loss_func_args)

        self._train_cache = self._init_cache()
        self._val_cache = self._init_cache()
        self._val_avg_loss = tm.MeanMetric()

    def _init_cache(self) -> CacheDict:
        cache = {}
        for k in self._cache_keys:
            cache[k] = []
            cache[f"{k}_sorted"] = []
        return cache

    def _build_loss_func(self, name: Literal["focal_loss", "l1_loss"], args: dict[str, Any]) -> nn.Module:
        if name == "focal_loss":
            return FocalLoss(**args)
        elif name == "l1_loss":
            return nn.L1Loss(**args)
        else:
            raise ValueError(f"Unsupported loss function: {name}")

    def _should_cache_train(self) -> bool:
        return self.trainer.current_epoch % self.trainer.check_val_every_n_epoch == 0

    @property
    def train_cache(self) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        return self._train_cache

    @property
    def val_cache(self) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        return self._val_cache

    @property
    def _cache_keys(self) -> list[str]:
        return [
            CacheKey.OUTPUT,
            CacheKey.LABEL_TS,
            CacheKey.LABEL_PRICE_OPEN,
            CacheKey.LABEL_PRICE_CLOSE,
            CacheKey.DECISION,
        ]

    def _compute_loss(self, output: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return self._loss_func(output, label)

    def _prepare_input(self, input_batch: torch.Tensor) -> torch.Tensor:
        return input_batch

    def _prepare_label(self, label_batch: torch.Tensor) -> torch.Tensor:
        return label_batch

    def _forward_pass(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self._network(x), dim=1)

    def _maybe_cache(
        self,
        cache: CacheDict | None,
        output: torch.Tensor,
        label_ts: torch.Tensor,
        label_price_open: torch.Tensor,
        label_price_close: torch.Tensor,
    ) -> None:
        if cache is not None:
            cache[CacheKey.OUTPUT].append(output)
            cache[CacheKey.LABEL_TS].append(label_ts)
            cache[CacheKey.LABEL_PRICE_OPEN].append(label_price_open)
            cache[CacheKey.LABEL_PRICE_CLOSE].append(label_price_close)

    def _clear_cache(self, cache: CacheDict) -> None:
        cache.update({k: [] for k in cache})

    def _step(self, batch, cache: CacheDict | None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        (
            x,
            y,
            _,
            y_ts,
            y_price_open,
            y_price_close,
        ) = batch

        x = self._prepare_input(x)
        y = self._prepare_label(y)

        output = self._forward_pass(x)
        loss = self._compute_loss(output, y)
        self._maybe_cache(cache, output, y_ts, y_price_open, y_price_close)

        return loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        cache = self._train_cache if self._should_cache_train() else None
        loss = self._step(batch, cache)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        loss = self._step(batch, self._val_cache)
        self._val_avg_loss(loss)
        self.log("val_loss", self._val_avg_loss, on_step=True, on_epoch=True)

    def configure_optimizers(self) -> Optimizer:
        optim = torch.optim.Adam(self._network.parameters(), lr=self._lr)
        return optim

    def on_train_epoch_start(self) -> None:
        self._clear_cache(self._train_cache)

    def on_validation_epoch_start(self) -> None:
        self._clear_cache(self._val_cache)

    def _process_cache(
        self,
        cache: CacheDict,
        output_key: str = CacheKey.OUTPUT,
        ts_key: str = CacheKey.LABEL_TS,
    ) -> None:
        for k, v in cache.items():
            if isinstance(v, list):
                cache[k] = torch.concat(cache[k])

        cache[CacheKey.DECISION] = self._output_to_decision(cache[output_key])

        sorted_key = ts_key + "_sorted"
        cache[sorted_key], sort_idx = cache[ts_key].flatten().sort()
        for key in cache.keys() - {sorted_key}:
            cache[key + "_sorted"] = cache[key].flatten()[sort_idx]

    def on_validation_epoch_end(self) -> None:
        if self.trainer.current_epoch > 0:
            self._process_cache(self._train_cache)
            self._process_cache(self._val_cache)

    def _output_to_decision(self, output: torch.Tensor) -> torch.Tensor:
        return output[:, -1] - output[:, 0]
