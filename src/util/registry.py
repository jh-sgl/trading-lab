import lightning as L
import pandas as pd
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import Dataset

DATAMODULE_REGISTRY = {}
DATASET_REGISTRY = {}
MODEL_REGISTRY = {}
NETWORK_REGISTRY = {}
LOSS_FUNC_REGISTRY = {}
CALLBACK_REGISTRY = {}


def register_datamodule(name):
    def wrapper(cls):
        if name in DATAMODULE_REGISTRY:
            raise ValueError(f"Duplicate dataset registry key: {name}")
        DATAMODULE_REGISTRY[name.lower()] = cls
        return cls

    return wrapper


def register_dataset(name):
    def wrapper(cls):
        if name in DATASET_REGISTRY:
            raise ValueError(f"Duplicate dataset registry key: {name}")
        DATASET_REGISTRY[name.lower()] = cls
        return cls

    return wrapper


def register_model(name):
    def wrapper(cls):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Duplicate model registry key: {name}")
        MODEL_REGISTRY[name.lower()] = cls
        return cls

    return wrapper


def register_network(name):
    def wrapper(cls):
        if name in NETWORK_REGISTRY:
            raise ValueError(f"Duplicate network registry key: {name}")
        NETWORK_REGISTRY[name.lower()] = cls
        return cls

    return wrapper


def register_loss_func(name):
    def wrapper(cls):
        if name in LOSS_FUNC_REGISTRY:
            raise ValueError(f"Duplicate loss func registry key: {name}")
        LOSS_FUNC_REGISTRY[name.lower()] = cls
        return cls

    return wrapper


def register_callback(name):
    def wrapper(cls):
        if name in CALLBACK_REGISTRY:
            raise ValueError(f"Duplicate callback registry key: {name}")
        CALLBACK_REGISTRY[name.lower()] = cls
        return cls

    return wrapper


def build_datamodule(datamodule_cfg: DictConfig) -> L.LightningDataModule:
    return DATAMODULE_REGISTRY[datamodule_cfg.name.lower()](**datamodule_cfg.args)


def build_dataset(dataset_cfg: DictConfig) -> Dataset:
    return DATASET_REGISTRY[dataset_cfg.name.lower()](**dataset_cfg.args)


def build_model(model_cfg: DictConfig) -> L.LightningModule:
    cls_ = MODEL_REGISTRY[model_cfg.name.lower()]
    if "checkpoint_fp" in model_cfg and model_cfg.checkpoint_fp is not None:
        return cls_.load_from_checkpoint(model_cfg.checkpoint_fp)
    else:
        return cls_(**model_cfg.args)


def build_network(network_cfg: DictConfig) -> nn.Module:
    return NETWORK_REGISTRY[network_cfg.name.lower()](**network_cfg.args)


def build_loss_func(loss_cfg: DictConfig) -> nn.Module:
    return LOSS_FUNC_REGISTRY[loss_cfg.name.lower()](**loss_cfg.args)


def build_callbacks(callback_cfg_list: list[DictConfig]) -> list[L.Callback]:
    return [CALLBACK_REGISTRY[callback_cfg.name.lower()](**callback_cfg.args) for callback_cfg in callback_cfg_list]


from task import *
