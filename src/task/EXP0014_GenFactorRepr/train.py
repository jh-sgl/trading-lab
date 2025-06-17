import os

import hydra
import lightning as L
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig

from .util.omegaconf import register_custom_resolver
from .util.prog_bar import ProgressBar
from .util.registry import build_callbacks, build_datamodule, build_model


@hydra.main(version_base=None, config_path="config", config_name=None)
def main(cfg: DictConfig) -> None:
    register_custom_resolver()

    output_dir = HydraConfig.get().runtime.output_dir
    logger = TensorBoardLogger(save_dir=output_dir, name="", version="")

    repr_model = build_model(cfg.repr.model)
    datamodule = build_datamodule(cfg.datamodule)
    datamodule.set_learning_stage("repr")
    repr_callbacks = build_callbacks(cfg.repr.callbacks)

    repr_model_checkpoint = ModelCheckpoint(
        dirpath=os.path.join(output_dir, "repr"),
        monitor="val_avg_loss",
        mode="min",
        save_top_k=1,
        filename="repr_best-{epoch:04d}-{val_avg_loss:.3f}",
        save_weights_only=False,
    )
    repr_callbacks.append(
        ProgressBar(desc=f"{os.path.basename(output_dir).split('/')[-1]}")
    )
    repr_callbacks.append(repr_model_checkpoint)
    repr_trainer = L.Trainer(
        devices=1,
        accelerator="gpu",
        callbacks=repr_callbacks,
        logger=logger,
        **cfg.repr.trainer.args,
    )
    repr_trainer.fit(model=repr_model, datamodule=datamodule)

    # TODO: train for MLPDropout after finishing representation learning
    best_repr_model = repr_model.__class__.load_from_checkpoint(
        repr_model_checkpoint.best_model_path
    )
    best_repr_model.eval()
    signal_model = build_model(cfg.signal.model)
    signal_model.set_repr_model(best_repr_model)
    datamodule.set_learning_stage("signal")
    signal_callbacks = build_callbacks(cfg.signal.callbacks)

    signal_model_checkpoint = ModelCheckpoint(
        dirpath=os.path.join(output_dir, "signal"),
        monitor="val_avg_loss",
        mode="min",
        save_top_k=1,
        filename="signal_best-{epoch:04d}-{val_avg_loss:.3f}",
        save_weights_only=False,
    )
    signal_callbacks.append(
        ProgressBar(desc=f"{os.path.basename(output_dir).split('/')[-1]}")
    )
    signal_callbacks.append(signal_model_checkpoint)
    signal_trainer = L.Trainer(
        devices=1,
        accelerator="gpu",
        callbacks=signal_callbacks,
        logger=logger,
        **cfg.signal.trainer.args,
    )
    signal_trainer.fit(model=signal_model, datamodule=datamodule)


if __name__ == "__main__":
    main()
