from typing import Any

import lightning as L
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from util.registry import build_dataset, register_datamodule


@register_datamodule("Basic")
class BasicDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_dataset: DictConfig,
        train_batch_size: int,
        train_num_workers: int,
        val_dataset: DictConfig,
        val_batch_size: int,
        val_num_workers: int,
    ) -> None:
        super().__init__()

        self._train_dataset = build_dataset(train_dataset)
        self._train_batch_size = train_batch_size
        self._train_num_workers = train_num_workers

        self._val_dataset = build_dataset(val_dataset)
        self._val_batch_size = val_batch_size
        self._val_num_workers = val_num_workers

    def train_dataloader(self) -> Any:
        return DataLoader(
            self._train_dataset,
            batch_size=self._train_batch_size,
            num_workers=self._train_num_workers,
        )

    def val_dataloader(self) -> Any:
        return DataLoader(
            self._val_dataset,
            batch_size=self._val_batch_size,
            num_workers=self._val_num_workers,
            shuffle=False,
        )
