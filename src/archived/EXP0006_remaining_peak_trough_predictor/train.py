import os

import hydra
import lightning as L
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig

from util.omegaconf import register_custom_resolver
from util.prog_bar import ProgressBar
from util.registry import build_callbacks, build_datamodule, build_model


@hydra.main(version_base=None, config_path="config", config_name="v1")
def main(cfg: DictConfig) -> None:
    register_custom_resolver()
    # TODO: modify model
    # TODO: implement backtester
    # TODO: analyze results (what happened near some specific days?)
    model = build_model(cfg.model)
    datamodule = build_datamodule(cfg.datamodule)
    callbacks = build_callbacks(cfg.callback)

    output_dir = HydraConfig.get().runtime.output_dir
    logger = TensorBoardLogger(save_dir=output_dir, name="", version="")

    callbacks.append(ProgressBar(desc=f"{os.path.basename(output_dir).split('/')[-1]}"))
    trainer = L.Trainer(devices=1, accelerator="gpu", callbacks=callbacks, logger=logger, **cfg.trainer.args)
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
