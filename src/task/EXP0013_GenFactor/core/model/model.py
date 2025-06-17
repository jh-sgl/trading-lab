import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchmetrics as tm
from omegaconf import DictConfig

from ...core.model.optim import SAM
from ...util.const import DFKey
from ...util.registry import build_loss_func, build_network, register_model


@register_model("basic")
class BasicModel(L.LightningModule):
    def __init__(
        self,
        network: DictConfig,
        loss_func: DictConfig,
        lr: float,
        optimizer: str,
        lr_scheduler: str,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self._network = build_network(network)
        self._loss_func = build_loss_func(loss_func)
        self._lr = lr

        self._train_avg_loss = tm.MeanMetric()
        self._val_avg_loss = tm.MeanMetric()

        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler

        if self._use_ASAM():
            self.automatic_optimization = False

    def on_fit_start(self) -> None:
        df_to_save = self.trainer.datamodule.df[self.trainer.datamodule.factor_cols]

        # handle duplicated columns
        seen = {}
        new_columns = []
        for col in df_to_save.columns:
            if col in seen:
                seen[col] += 1
                new_col = f"{col}_dup{seen[col]}"
            else:
                seen[col] = 0
                new_col = col
            new_columns.append(new_col)
        df_to_save.columns = pd.MultiIndex.from_tuples(new_columns)
        df_to_save.to_parquet(f"{self.trainer.log_dir}/factors.parquet")

    def _use_ASAM(self) -> bool:
        return self._optimizer == "ASAM"

    @property
    def network(self) -> nn.Module:
        return self._network

    def configure_optimizers(self):
        optim_dict = {}
        if self._optimizer == "ASAM":
            optim_dict["optimizer"] = SAM(
                self._network.parameters(),
                base_optimizer=torch.optim.SGD,
                rho=0.2,
                adaptive=True,
                lr=self._lr,
                momentum=0.9,
            )
        elif self._optimizer == "AdamW":
            optim_dict["optimizer"] = torch.optim.AdamW(
                self._network.parameters(), lr=self._lr
            )
        else:
            raise ValueError(f"Unknown optimizer: {self._optimizer}")

        if self._lr_scheduler == "CosineAnnealingWarmRestarts":
            optim_dict["lr_scheduler"] = {
                "scheduler": CosineAnnealingWarmRestarts(
                    optim_dict["optimizer"], T_0=10, T_mult=2
                ),
                "interval": "epoch",
                "frequency": 1,
            }
        elif self._lr_scheduler is None:
            pass

        return optim_dict

    def _compute_loss(
        self, outputs: dict[DFKey, torch.Tensor], labels: dict[DFKey, torch.Tensor]
    ) -> torch.Tensor:
        loss = self._loss_func(outputs, labels)
        return loss

    def _step(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        inputs, label, _ = batch
        outputs = self._network(inputs)
        loss = self._compute_loss(outputs, label)
        return outputs, loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        outputs, loss = self._step(batch)

        if self._use_ASAM():
            opt = self.optimizers()
            self.manual_backward(loss)
            with torch.no_grad():
                opt.first_step(zero_grad=True)
                with torch.set_grad_enabled(True):
                    outputs_perturbed, loss_perturbed = self._step(batch)
                    self.manual_backward(loss_perturbed)
                opt.second_step()

        self._train_avg_loss(loss)
        self.log("train_loss", self._train_avg_loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        outputs, loss = self._step(batch)
        self._val_avg_loss(loss)
        self.log("val_loss", self._val_avg_loss, on_epoch=True, prog_bar=True)
        timestamp = batch[-1]
        self._store_outputs_to_df(self.trainer.datamodule.df, outputs, timestamp)

    def _store_outputs_to_df(
        self, df: pd.DataFrame, outputs: torch.Tensor, timeindex: torch.Tensor
    ) -> None:
        outputs = [o.tolist() for o in outputs.detach().cpu().numpy()]
        timeindex = timeindex.detach().cpu().numpy()

        output_df = pd.DataFrame(
            {
                DFKey.OUTPUT_PRED_SHORT: [o[0] for o in outputs],
                DFKey.OUTPUT_PRED_HOLD: [o[1] for o in outputs],
                DFKey.OUTPUT_PRED_LONG: [o[2] for o in outputs],
                "timeindex": timeindex,
            }
        )
        output_df["timeindex"] = output_df["timeindex"].apply(
            lambda x: pd.to_datetime(x)
        )
        output_df = output_df.set_index("timeindex")
        output_df["timeindex"] = output_df.index

        df.loc[output_df.index, DFKey.OUTPUT_PRED_SHORT] = output_df[
            DFKey.OUTPUT_PRED_SHORT
        ]
        df.loc[output_df.index, DFKey.OUTPUT_PRED_HOLD] = output_df[
            DFKey.OUTPUT_PRED_HOLD
        ]
        df.loc[output_df.index, DFKey.OUTPUT_PRED_LONG] = output_df[
            DFKey.OUTPUT_PRED_LONG
        ]


@register_model("mixup")
class MixupModel(BasicModel):
    def __init__(
        self,
        network: DictConfig,
        loss_func: DictConfig,
        lr: float,
        optimizer: str,
        lr_scheduler: str,
        mixup_alpha: float,
    ) -> None:
        super().__init__(network, loss_func, lr, optimizer, lr_scheduler)
        self._mixup_alpha = mixup_alpha

    def _mixup(
        self, inputs: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        lambda_ = (
            np.random.beta(self._mixup_alpha, self._mixup_alpha)
            if self._mixup_alpha > 0.0
            else 1.0
        )
        index = torch.randperm(inputs.size()[0]).to(inputs.device)
        inputs_mixup = lambda_ * inputs + (1 - lambda_) * inputs[index]
        labels_a, labels_b = labels, labels[index]
        return inputs_mixup, labels_a, labels_b, lambda_

    def _compute_loss(
        self,
        outputs: torch.Tensor,
        labels_a: torch.Tensor,
        labels_b: torch.Tensor,
        lambda_: float,
    ) -> torch.Tensor:
        loss_a = self._loss_func(outputs, labels_a)
        loss_b = self._loss_func(outputs, labels_b)
        loss = lambda_ * loss_a + (1 - lambda_) * loss_b
        return loss

    def _step(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        inputs, labels, _ = batch

        inputs_mixup, labels_a, labels_b, lambda_ = self._mixup(inputs, labels)
        outputs = self._network(inputs_mixup)
        loss = self._compute_loss(outputs, labels_a, labels_b, lambda_)
        return outputs, loss
