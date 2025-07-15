from typing import Callable

import lightning as L
import numpy as np
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import Dataset

DATAMODULE_REGISTRY = {}
DATASET_REGISTRY = {}
MODEL_REGISTRY = {}
NETWORK_REGISTRY = {}
LOSS_FUNC_REGISTRY = {}
CALLBACK_REGISTRY = {}
BACKTESTER_REGISTRY = {}
STOP_LOSS_FUNC_REGISTRY = {}
TAKE_PROFIT_FUNC_REGISTRY = {}
GENFACTOR_REGISTRY = {}


def register_datamodule(name):
    name = name.lower()

    def wrapper(cls):
        if name in DATAMODULE_REGISTRY:
            raise ValueError(f"Duplicate dataset registry key: {name}")
        DATAMODULE_REGISTRY[name] = cls
        return cls

    return wrapper


def register_dataset(name):
    name = name.lower()

    def wrapper(cls):
        if name in DATASET_REGISTRY:
            raise ValueError(f"Duplicate dataset registry key: {name}")
        DATASET_REGISTRY[name] = cls
        return cls

    return wrapper


def register_model(name):
    name = name.lower()

    def wrapper(cls):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Duplicate model registry key: {name}")
        MODEL_REGISTRY[name] = cls
        return cls

    return wrapper


def register_network(name):
    name = name.lower()

    def wrapper(cls):
        if name in NETWORK_REGISTRY:
            raise ValueError(f"Duplicate network registry key: {name}")
        NETWORK_REGISTRY[name] = cls
        return cls

    return wrapper


def register_loss_func(name):
    name = name.lower()

    def wrapper(cls):
        if name in LOSS_FUNC_REGISTRY:
            raise ValueError(f"Duplicate loss func registry key: {name}")
        LOSS_FUNC_REGISTRY[name] = cls
        return cls

    return wrapper


def register_callback(name):
    name = name.lower()

    def wrapper(cls):
        if name in CALLBACK_REGISTRY:
            raise ValueError(f"Duplicate callback registry key: {name}")
        CALLBACK_REGISTRY[name] = cls
        return cls

    return wrapper


def register_backtester(name):
    name = name.lower()

    def wrapper(cls):
        if name in BACKTESTER_REGISTRY:
            raise ValueError(f"Duplicate backtester registry key: {name}")
        BACKTESTER_REGISTRY[name] = cls
        return cls

    return wrapper


def register_stop_loss_func(name):
    name = name.lower()

    def wrapper(factory_func):
        if name in STOP_LOSS_FUNC_REGISTRY:
            raise ValueError(f"Duplicate stop loss func registry key: {name}")
        STOP_LOSS_FUNC_REGISTRY[name] = factory_func
        return factory_func

    return wrapper


def register_take_profit_func(name):
    name = name.lower()

    def wrapper(factory_func):
        if name in TAKE_PROFIT_FUNC_REGISTRY:
            raise ValueError(f"Duplicate take profit func registry key: {name}")
        TAKE_PROFIT_FUNC_REGISTRY[name] = factory_func
        return factory_func

    return wrapper


def register_genfactor(name):
    name = name.lower()

    def wrapper(cls):
        if name in GENFACTOR_REGISTRY:
            raise ValueError(f"Duplicate genfactor registry key: {name}")
        GENFACTOR_REGISTRY[name] = cls
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


def build_network(network_cfg: DictConfig, **kwargs) -> nn.Module:
    return NETWORK_REGISTRY[network_cfg.name.lower()](**network_cfg.args, **kwargs)


def build_loss_func(loss_cfg: DictConfig) -> nn.Module:
    return LOSS_FUNC_REGISTRY[loss_cfg.name.lower()](**loss_cfg.args)


def build_callbacks(callback_cfg_list: list[DictConfig]) -> list[L.Callback]:
    return [
        CALLBACK_REGISTRY[callback_cfg.name.lower()](**callback_cfg.args)
        for callback_cfg in callback_cfg_list
    ]


def build_backtester(backtester_cfg: DictConfig) -> "Backtester":
    return BACKTESTER_REGISTRY[backtester_cfg.name.lower()](**backtester_cfg.args)


def build_stop_loss_func(
    stop_loss_cfg: DictConfig,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    name = stop_loss_cfg.name.lower()
    kwargs = {k: v for k, v in stop_loss_cfg.args.items()}
    return STOP_LOSS_FUNC_REGISTRY[name](**kwargs)


def build_take_profit_func(
    take_profit_cfg: DictConfig,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    name = take_profit_cfg.name.lower()
    kwargs = {k: v for k, v in take_profit_cfg.args.items()}
    return TAKE_PROFIT_FUNC_REGISTRY[name](**kwargs)


def build_genfactor(genfactor_cfg: DictConfig) -> "GenFactor":
    name = genfactor_cfg.name.lower()
    if genfactor_cfg.args == "random":
        return GENFACTOR_REGISTRY[name](random_params=True)
    else:
        kwargs = {k: v for k, v in genfactor_cfg.args.items()}
        return GENFACTOR_REGISTRY[name](**kwargs)


from ..core import *
