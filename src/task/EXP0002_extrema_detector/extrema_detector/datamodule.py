from typing import Any

import lightning as L
import pandas as pd
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from util.registry import build_dataset, register_datamodule


@register_datamodule("ExtremaDetector")
class ExtremaDetectorDataModule(L.LightningDataModule):
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

        self._train_dataloader = None
        self._val_dataloader = None

    @property
    def train_dataframe(self) -> pd.DataFrame:
        return self._train_dataset.df

    @property
    def val_dataframe(self) -> pd.DataFrame:
        return self._val_dataset.df

    def train_dataloader(self) -> Any:
        if self._train_dataloader is None:
            self._train_dataloader = DataLoader(
                self._train_dataset,
                batch_size=self._train_batch_size,
                num_workers=self._train_num_workers,
                prefetch_factor=self._train_num_workers * 2 if self._train_num_workers > 0 else None,
                persistent_workers=True if self._train_num_workers > 0 else False,
                pin_memory=True,
            )
        return self._train_dataloader

    def val_dataloader(self) -> Any:
        if self._val_dataloader is None:
            self._val_dataloader = DataLoader(
                self._val_dataset,
                batch_size=self._val_batch_size,
                num_workers=self._val_num_workers,
                prefetch_factor=self._val_num_workers * 2 if self._val_num_workers > 0 else None,
                shuffle=False,
                persistent_workers=True if self._val_num_workers > 0 else False,
                pin_memory=True,
            )
        return self._val_dataloader
