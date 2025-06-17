import lightning as L
import torch
import torchmetrics as tm
from omegaconf import DictConfig
from torch.optim import Optimizer

from util.registry import build_loss_func, build_network, register_model

from .const import CacheDict, DataKey


@register_model("ExtremaDetector")
class ExtremaDetectorModel(L.LightningModule):
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
        return [DataKey.OUTPUT_PRED, DataKey.TIMESTAMP]

    def _compute_loss(self, output: torch.Tensor, price_exit_label: torch.Tensor) -> torch.Tensor:
        return self._loss_func(output, price_exit_label)

    def _prepare_input(self, input_batch: torch.Tensor) -> torch.Tensor:
        return input_batch

    def _prepare_label(self, label_batch: torch.Tensor) -> torch.Tensor:
        return label_batch

    def _forward_pass(self, candle_inputs: torch.Tensor) -> torch.Tensor:
        return self._network(candle_inputs)

    def _maybe_cache(
        self,
        cache: CacheDict | None,
        output_pred: torch.Tensor,
        timestamp: torch.Tensor,
    ) -> None:
        if cache is not None:
            cache[DataKey.OUTPUT_PRED].append(output_pred)
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
        elif len(output.shape) > 1:
            output = output.mean(dim=-1)

        if not self.training and self.current_epoch > 25:
            for o, p in zip(output, price_exit_label):
                print(f"{(o - p).item():.2f} / {o.item():.2f} / {p.item():.2f}")

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
        optim = torch.optim.Adam(self._network.parameters(), lr=self._lr)
        return optim

    def on_train_epoch_start(self) -> None:
        self._train_cache = self._init_cache()

    def on_validation_epoch_start(self) -> None:
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
