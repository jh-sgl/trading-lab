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

    datamodule = build_datamodule(cfg.datamodule)
    datamodule.set_learning_stage("signal_wo_repr")

    signal_model = build_model(cfg.signal.model)

    signal_callbacks = build_callbacks(cfg.signal.callbacks)
    signal_model_checkpoint = ModelCheckpoint(
        dirpath=os.path.join(output_dir, "signal_wo_repr"),
        monitor="val_avg_loss",
        mode="min",
        save_top_k=1,
        filename="signal_wo_repr_best-{epoch:04d}-{val_avg_loss:.3f}",
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
