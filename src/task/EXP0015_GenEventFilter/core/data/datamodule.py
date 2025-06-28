from typing import Any, Literal

import lightning as L
import pandas as pd
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset

from ...util.registry import build_dataset, register_datamodule
from .dataset import BasicDataset


@register_datamodule("basic_datamodule")
class BasicDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset: DictConfig,
        batch_size: int,
        num_workers: int,
    ) -> None:
        super().__init__()

        self._learning_stage: Literal["repr", "signal", "signal_wo_repr"]
        self._dataset: BasicDataset = build_dataset(dataset)

        self._repr_train_dataset: Subset[BasicDataset] = Subset(
            self._dataset, self._dataset.repr_train_dataloader_idx
        )
        self._repr_valid_dataset: Subset[BasicDataset] = Subset(
            self._dataset, self._dataset.repr_valid_dataloader_idx
        )
        self._signal_train_dataset: Subset[BasicDataset] = Subset(
            self._dataset, self._dataset.signal_train_dataloader_idx
        )
        self._signal_total_dataset: Subset[BasicDataset] = Subset(
            self._dataset, self._dataset.signal_total_dataloader_idx
        )

        self._batch_size = batch_size
        self._num_workers = num_workers

        self._learning_stage = "repr"

    @property
    def df(self) -> pd.DataFrame:
        return self._dataset.df

    @property
    def factor_cols(self) -> pd.MultiIndex:
        return self._dataset.factor_cols

    @property
    def input_dim(self) -> int:
        return len(self._dataset.factor_cols)

    @property
    def geneventfilter_name(self) -> pd.MultiIndex:
        return self._dataset.geneventfilter_name

    def set_learning_stage(
        self, learning_stage: Literal["repr", "signal", "signal_wo_repr"]
    ) -> None:
        self._learning_stage = learning_stage
        self._train_dataloader = None
        self._valid_dataloader = None
        self._dataset.set_learning_stage(self._learning_stage)

    def train_dataloader(self) -> Any:
        if self._learning_stage == "repr":
            dataset = self._repr_train_dataset
        elif self._learning_stage in ["signal", "signal_wo_repr"]:
            dataset = self._signal_train_dataset

        if self._train_dataloader is None:
            self._train_dataloader = DataLoader(
                dataset,
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
        if self._learning_stage == "repr":
            dataset = self._repr_valid_dataset
        elif self._learning_stage in ["signal", "signal_wo_repr"]:
            dataset = self._signal_total_dataset

        if self._valid_dataloader is None:
            self._valid_dataloader = DataLoader(
                dataset,
                batch_size=self._batch_size,
                num_workers=self._num_workers,
                prefetch_factor=(
                    self._num_workers * 2 if self._num_workers > 0 else None
                ),
                shuffle=False,
                persistent_workers=True if self._num_workers > 0 else False,
                pin_memory=True,
            )
        return self._valid_dataloader
