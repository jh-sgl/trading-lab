import lightning as L
import torch
import torchmetrics as tm
from omegaconf import DictConfig
from torch.optim import Optimizer

from util.registry import build_loss_func, register_model

from .const import CacheDict, CacheKey
from .dataset import DataItem, MetaInfo, PastInputs, TodayLabels
from .network import DLinear, DLinearV2


@register_model("BandPredictor")
class BandPredictorModel(L.LightningModule):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        input_ch: int,
        output_num: int,
        moving_avg_kernel_size: int,
        lookback_days: int,
        lr: float,
        loss_func: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self._network = DLinearV2(seq_len, pred_len, input_ch, output_num, lookback_days, moving_avg_kernel_size)
        self._lr = lr
        self._loss_func = build_loss_func(loss_func)

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
        return [CacheKey.TS_LABEL, CacheKey.BAND_OFFSET, CacheKey.BAND_CENTER_PRED, CacheKey.TODAY_CUTOFF_MEAN_PRICE]

    def _compute_loss(self, output: torch.Tensor, today_labels: TodayLabels) -> torch.Tensor:
        return self._loss_func(output, today_labels.today_band_center)

    def _prepare_input(self, input_batch: torch.Tensor) -> torch.Tensor:
        return input_batch

    def _prepare_label(self, label_batch: torch.Tensor) -> torch.Tensor:
        return label_batch

    def _forward_pass(self, past_inputs: PastInputs) -> torch.Tensor:
        return self._network(past_inputs)

    def _maybe_cache(
        self,
        cache: CacheDict | None,
        ts_label: torch.Tensor,
        band_offset: torch.Tensor,
        today_cutoff_mean_price: torch.Tensor,
        band_center_pred: torch.Tensor,
    ) -> None:
        if cache is not None:
            cache[CacheKey.TS_LABEL].append(ts_label)
            cache[CacheKey.BAND_OFFSET].append(band_offset)
            cache[CacheKey.BAND_CENTER_PRED].append(band_center_pred)
            cache[CacheKey.TODAY_CUTOFF_MEAN_PRICE].append(today_cutoff_mean_price)

    def _batch_to_data_item(self, batch) -> DataItem:
        data_item = DataItem(
            past_inputs=PastInputs(**batch["past_inputs"]),
            today_labels=TodayLabels(**batch["today_labels"]),
            meta_info=MetaInfo(**batch["meta_info"]),
        )
        return data_item

    def _step(self, batch, cache: CacheDict | None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        data_item = self._batch_to_data_item(batch)
        output = self._forward_pass(data_item.past_inputs)
        loss = self._compute_loss(output, data_item.today_labels)

        output = output.mean(dim=-1) if len(output.shape) > 1 else output
        if not self.training:
            for o, c in zip(output, data_item.today_labels.today_band_center):
                print(f"{abs((o - c).item()):.2f} \t|\t{o.item():.2f}\t|\t{c.item():.2f}")

        meta_info = data_item.meta_info
        self._maybe_cache(
            cache, meta_info.today_timestamp, meta_info.today_band_offset, meta_info.today_cutoff_mean_price, output
        )
        return loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        cache = self._train_cache if self._should_cache_train() else None
        loss = self._step(batch, cache)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        loss = self._step(batch, self._val_cache)
        self._val_avg_loss(loss)
        self.log("val_loss", self._val_avg_loss, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> Optimizer:
        optim = torch.optim.Adam(self._network.parameters(), lr=self._lr)
        return optim

    def on_train_epoch_start(self) -> None:
        self._train_cache = self._init_cache()

    def on_validation_epoch_start(self) -> None:
        self._val_cache = self._init_cache()

    def _process_cache(self, cache: CacheDict, ts_key: str = CacheKey.TS_LABEL) -> None:
        for k, v in cache.items():
            if isinstance(v, list):
                cache[k] = torch.concat(cache[k])

        cache[ts_key], sort_idx = cache[ts_key].flatten().sort()
        for key in cache.keys() - {ts_key}:
            cache[key] = cache[key].flatten()[sort_idx]

    def on_validation_epoch_end(self) -> None:
        if self.trainer.sanity_checking:
            return

        self._process_cache(self._train_cache)
        self._process_cache(self._val_cache)

    def _output_to_decision(self, output: torch.Tensor) -> torch.Tensor:
        return output[:, -1] - output[:, 0]
