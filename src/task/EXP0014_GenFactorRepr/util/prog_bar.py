from lightning.pytorch.callbacks import TQDMProgressBar


class ProgressBar(TQDMProgressBar):
    def __init__(self, desc: str) -> None:
        super().__init__()
        self._desc = desc

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ) -> None:
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        self.val_progress_bar.set_description(
            desc=f"[{self._desc}] E{trainer.current_epoch} {self.validation_description}".replace("Validation", "VALID")
        )
        return

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ) -> None:
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        self.train_progress_bar.set_description(
            desc=f"[{self._desc}] E{trainer.current_epoch} {self.train_description}".replace("Training", "TRAIN")
        )
        return
