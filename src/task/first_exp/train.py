from datetime import datetime

import hydra
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig

from util.omegaconf import register_custom_resolver
from util.registry import build_callbacks, build_datamodule, build_model


@hydra.main(version_base=None, config_path="config", config_name="v1")
def main(cfg: DictConfig) -> None:
    register_custom_resolver()

    model = build_model(cfg.model)
    datamodule = build_datamodule(cfg.datamodule)
    callbacks = build_callbacks(cfg.callback)

    logger = TensorBoardLogger(
        save_dir="logs", name=f"{datetime.now().strftime('%Y-%m-%d')}", version=datetime.now().strftime("%H_%M_%S")
    )

    trainer = L.Trainer(devices=1, accelerator="gpu", callbacks=callbacks, logger=logger, **cfg.trainer.args)
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
