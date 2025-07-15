import logging
from typing import Any, Literal

import lightning as L
import pandas as pd
import torch
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
        self._regime_train_dataset: Subset[BasicDataset] = Subset(
            self._dataset, self._dataset.regime_train_dataloader_idx
        )
        self._regime_valid_dataset: Subset[BasicDataset] = Subset(
            self._dataset, self._dataset.regime_valid_dataloader_idx
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
    def factor_tensor(self) -> torch.Tensor:
        return self._dataset.factor_tensor

    @property
    def repr_lookback_num(self) -> int:
        return self._dataset.repr_lookback_num

    @property
    def regime_num_classes(self) -> int:
        return self._dataset.regime_num_classes

    @property
    def signal_train_repr_tensor(self) -> torch.Tensor:
        return self._dataset.signal_train_repr_tensor

    @property
    def signal_train_label_tensor(self) -> torch.Tensor:
        return self._dataset.signal_train_label_tensor

    def set_learning_stage(
        self,
        learning_stage: Literal["repr", "regime", "signal", "signal_wo_repr"],
        repr_model: L.LightningModule | None = None,
    ) -> None:
        self._learning_stage = learning_stage
        self._train_dataloader = None
        self._valid_dataloader = None

        if learning_stage in ["regime", "signal"] and repr_model is not None:
            self._dataset.set_learning_stage(self._learning_stage, repr_model)

    def train_dataloader(self) -> Any:
        if self._learning_stage == "repr":
            dataset = self._repr_train_dataset
        elif self._learning_stage == "regime":
            dataset = self._regime_train_dataset
        elif self._learning_stage in ["signal", "signal_wo_repr"]:
            dataset = self._signal_train_dataset
        else:
            raise ValueError(f"Unknown learning_stage: {self._learning_stage}")

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
        elif self._learning_stage == "regime":
            dataset = self._regime_valid_dataset
        elif self._learning_stage in ["signal", "signal_wo_repr"]:
            dataset = self._signal_total_dataset
        else:
            raise ValueError(f"Unknown learning_stage: {self._learning_stage}")

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
