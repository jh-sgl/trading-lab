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

    # 1. TRAIN REPRESENTATION MODEL
    repr_model = build_model(cfg.repr.model)
    datamodule = build_datamodule(cfg.datamodule)
    datamodule.set_learning_stage("repr")
    repr_callbacks = build_callbacks(cfg.repr.callbacks)

    repr_model_checkpoint = ModelCheckpoint(
        dirpath=os.path.join(output_dir, "repr"),
        monitor="repr_val_avg_loss",
        mode="min",
        save_top_k=1,
        filename="repr_best-{epoch:04d}-{repr_val_avg_loss:.3f}",
        save_weights_only=False,
    )
    repr_callbacks.append(
        ProgressBar(desc=f"{os.path.basename(output_dir).split('/')[-1]}-REPR")
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

    best_repr_model = repr_model.__class__.load_from_checkpoint(
        repr_model_checkpoint.best_model_path
    )
    best_repr_model.eval()

    # 2. TRAIN SIGNAL MODEL
    regime_model = build_model(cfg.regime.model)
    datamodule.set_learning_stage("regime", best_repr_model)
    regime_callbacks = build_callbacks(cfg.regime.callbacks)

    regime_model_checkpoint = ModelCheckpoint(
        dirpath=os.path.join(output_dir, "regime"),
        monitor="regime_val_avg_loss",
        mode="min",
        save_top_k=1,
        filename="regime_best-{epoch:04d}-{regime_val_avg_loss:.3f}",
        every_n_epochs=1,
        save_weights_only=False,
    )
    regime_callbacks.append(
        ProgressBar(desc=f"{os.path.basename(output_dir).split('/')[-1]}-REGIME")
    )
    regime_callbacks.append(regime_model_checkpoint)
    regime_trainer = L.Trainer(
        devices=1,
        accelerator="gpu",
        callbacks=regime_callbacks,
        logger=logger,
        **cfg.regime.trainer.args,
    )
    regime_trainer.fit(model=regime_model, datamodule=datamodule)

    # 3. TRAIN SIGNAL MODEL
    signal_model = build_model(cfg.signal.model)
    datamodule.set_learning_stage("signal", best_repr_model)
    signal_callbacks = build_callbacks(cfg.signal.callbacks)

    signal_model_checkpoint = ModelCheckpoint(
        dirpath=os.path.join(output_dir, "signal"),
        monitor="signal_val_avg_loss",
        mode="min",
        save_top_k=1,
        filename="signal_best-{epoch:04d}-{signal_val_avg_loss:.3f}",
        every_n_epochs=1,
        save_weights_only=False,
    )
    signal_callbacks.append(
        ProgressBar(desc=f"{os.path.basename(output_dir).split('/')[-1]}-SIGNAL")
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
