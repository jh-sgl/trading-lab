from typing import Any

import lightning as L
import pandas as pd
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset

from ...util.registry import build_dataset, register_datamodule
from .dataset import BasicDataset


@register_datamodule("basic")
class BasicDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset: DictConfig,
        batch_size: int,
        num_workers: int,
    ) -> None:
        super().__init__()

        self._dataset: BasicDataset = build_dataset(dataset)
        self._train_dataset: Subset[BasicDataset] = Subset(
            self._dataset, self._dataset.train_dataloader_idx
        )
        self._total_dataset: Subset[BasicDataset] = Subset(
            self._dataset, self._dataset.total_dataloader_idx
        )

        self._batch_size = batch_size
        self._num_workers = num_workers

        self._train_dataloader = None
        self._total_dataloader = None

    @property
    def df(self) -> pd.DataFrame:
        return self._dataset.df

    @property
    def factor_cols(self) -> pd.MultiIndex:
        return self._dataset.factor_cols

    @property
    def train_date_range(self) -> tuple[str, str]:
        return self._dataset.train_date_range

    @property
    def total_date_range(self) -> tuple[str, str]:
        return self._dataset.total_date_range

    def train_dataloader(self) -> Any:
        if self._train_dataloader is None:
            self._train_dataloader = DataLoader(
                self._train_dataset,
                batch_size=self._batch_size,
                num_workers=self._num_workers,
                prefetch_factor=(
                    self._num_workers * 2 if self._num_workers > 0 else None
                ),
                persistent_workers=True if self._num_workers > 0 else False,
                pin_memory=True,
                shuffle=True,
            )
        return self._train_dataloader

    def val_dataloader(self) -> Any:
        if self._total_dataloader is None:
            self._total_dataloader = DataLoader(
                self._total_dataset,
                batch_size=self._batch_size,
                num_workers=self._num_workers,
                prefetch_factor=(
                    self._num_workers * 2 if self._num_workers > 0 else None
                ),
                shuffle=False,
                persistent_workers=True if self._num_workers > 0 else False,
                pin_memory=True,
            )
        return self._total_dataloader
