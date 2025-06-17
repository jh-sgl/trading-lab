import logging

import lightning as L
import torch
import torchmetrics as tm
from omegaconf import DictConfig
from torch.optim import Optimizer

from util.registry import build_loss_func, build_network, register_model

from .const import CacheDict, DataKey


@register_model("RemainingPeakTroughPredictor")
class RemainingPeakTroughPredictorModel(L.LightningModule):
    def __init__(self, network: DictConfig, lr: float, loss_func: DictConfig, use_soft_label: bool) -> None:
        super().__init__()
        self.save_hyperparameters()

        self._network = build_network(network)
        self._lr = lr
        self._loss_func = build_loss_func(loss_func)
        self._use_soft_label = use_soft_label

        self._train_cache = self._init_cache()
        self._val_cache = self._init_cache()
        self._val_avg_loss = tm.MeanMetric()

    def _init_cache(self) -> CacheDict:
        cache = {}
        for k in self._cache_keys:
            cache[k] = []
        return cache

    def _should_cache_train(self) -> bool:
        return (self.trainer.current_epoch + 1) % self.trainer.check_val_every_n_epoch == 0 and self.training

    @property
    def train_cache(self) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        return self._train_cache

    @property
    def val_cache(self) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        return self._val_cache

    @property
    def _cache_keys(self) -> list[str]:
        return [DataKey.TIMESTAMP, DataKey.OUTPUT_PRED_BUY_NORMALIZED, DataKey.OUTPUT_PRED_SELL_NORMALIZED]

    def _compute_loss(self, output: torch.Tensor, price_exit_label: torch.Tensor) -> torch.Tensor:
        return self._loss_func(output, price_exit_label)

    def _prepare_input(self, input_batch: torch.Tensor) -> torch.Tensor:
        return input_batch

    def _prepare_label(self, label_batch: torch.Tensor) -> torch.Tensor:
        return label_batch

    def _forward_pass(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self._network(inputs)
        if self._use_soft_label:
            output = output.view(output.shape[0], 2, output.shape[-1] // 2)
        return output

    def _maybe_cache(
        self,
        cache: CacheDict | None,
        output_pred: torch.Tensor,
        timestamp: torch.Tensor,
    ) -> None:
        if cache is not None:
            cache[DataKey.OUTPUT_PRED_BUY_NORMALIZED].append(output_pred[:, 0])
            cache[DataKey.OUTPUT_PRED_SELL_NORMALIZED].append(output_pred[:, 1])
            cache[DataKey.TIMESTAMP].append(timestamp)

    def _step(self, batch, cache: CacheDict | None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        candle_inputs, price_exit_label, timestamp = batch

        output = self._forward_pass(candle_inputs)
        if self._use_soft_label:
            output = torch.softmax(output, dim=-1)

        loss = self._compute_loss(output, price_exit_label)

        if self._use_soft_label:
            output = output[..., -1] - output[..., 0]
            price_exit_label = price_exit_label[..., -1] - price_exit_label[..., 0]

        if not self.training:
            buy_output, buy_label = output[0, 0], price_exit_label[0, 0]
            sell_output, sell_label = output[0, 1], price_exit_label[0, 1]
            self._print_item += (
                f"{(buy_output - buy_label).item():.2f} / {buy_output.item():.2f} / {buy_label.item():.2f}\n"
            )
            self._print_item += (
                f"{(sell_output - sell_label).item():.2f} / {sell_output.item():.2f} / {sell_label.item():.2f}\n"
            )

        self._maybe_cache(cache, output, timestamp)
        return loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        cache = self._train_cache if self._should_cache_train() else None
        loss = self._step(batch, cache)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        loss = self._step(batch, self._val_cache)
        self._val_avg_loss(loss)
        self.log("val_loss", self._val_avg_loss, on_epoch=True)

    def configure_optimizers(self) -> Optimizer:
        optim = torch.optim.AdamW(self._network.parameters(), lr=self._lr, weight_decay=0.01)
        return optim

    def on_train_epoch_start(self) -> None:
        self._train_cache = self._init_cache()

    def on_validation_epoch_start(self) -> None:
        self._print_item = ""
        self._val_cache = self._init_cache()

    def _process_cache(self, cache: CacheDict, ts_key: str = DataKey.TIMESTAMP) -> None:
        for k, v in cache.items():
            if isinstance(v, list):
                cache[k] = torch.concat(cache[k])

        cache[ts_key], sort_idx = cache[ts_key].flatten().sort()
        for key in cache.keys() - {ts_key}:
            cache[key] = cache[key].flatten(-1)[sort_idx]

    def on_validation_epoch_end(self) -> None:
        if self.trainer.sanity_checking:
            return

        self._process_cache(self._train_cache)
        self._process_cache(self._val_cache)
        logging.info(self._print_item)
