import lightning as L
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import Dataset

DATAMODULE_REGISTRY = {}
DATASET_REGISTRY = {}
MODEL_REGISTRY = {}
LOSS_FUNC_REGISTRY = {}
CALLBACK_REGISTRY = {}


def register_datamodule(name):
    def wrapper(cls):
        if name in DATAMODULE_REGISTRY:
            raise ValueError(f"Duplicate dataset registry key: {name}")
        DATAMODULE_REGISTRY[name] = cls
        return cls

    return wrapper


def register_dataset(name):
    def wrapper(cls):
        if name in DATASET_REGISTRY:
            raise ValueError(f"Duplicate dataset registry key: {name}")
        DATASET_REGISTRY[name] = cls
        return cls

    return wrapper


def register_model(name):
    def wrapper(cls):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Duplicate model registry key: {name}")
        MODEL_REGISTRY[name] = cls
        return cls

    return wrapper


def register_loss_func(name):
    def wrapper(cls):
        if name in LOSS_FUNC_REGISTRY:
            raise ValueError(f"Duplicate loss func registry key: {name}")
        LOSS_FUNC_REGISTRY[name] = cls
        return cls

    return wrapper


def register_callback(name):
    def wrapper(cls):
        if name in CALLBACK_REGISTRY:
            raise ValueError(f"Duplicate callback registry key: {name}")
        CALLBACK_REGISTRY[name] = cls
        return cls

    return wrapper


def build_datamodule(datamodule_cfg: DictConfig) -> L.LightningDataModule:
    return DATAMODULE_REGISTRY[datamodule_cfg.name](**datamodule_cfg.args)


def build_dataset(dataset_cfg: DictConfig) -> Dataset:
    return DATASET_REGISTRY[dataset_cfg.name](**dataset_cfg.args)


def build_model(model_cfg: DictConfig) -> L.LightningModule:
    return MODEL_REGISTRY[model_cfg.name](**model_cfg.args)


def build_loss_func(loss_cfg: DictConfig) -> nn.Module:
    return LOSS_FUNC_REGISTRY[loss_cfg.name](**loss_cfg.args)


def build_callbacks(callback_cfg_list: list[DictConfig]) -> list[L.Callback]:
    return [CALLBACK_REGISTRY[callback_cfg.name](**callback_cfg.args) for callback_cfg in callback_cfg_list]


from datamodule import *
from datamodule.dataset import *
from model import *
from model.common.loss import *
