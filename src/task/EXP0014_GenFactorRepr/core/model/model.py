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


@register_model("repr_model")
class ReprModel(L.LightningModule):
    def __init__(
        self,
        network: DictConfig,
        loss_func: DictConfig,
        lr: float,
        optimizer: str,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self._network = build_network(network)
        self._loss_func = build_loss_func(loss_func)
        self._lr = lr

        self._train_avg_loss = tm.MeanMetric()
        self._val_avg_loss = tm.MeanMetric()

        self._optimizer = optimizer

    @property
    def network(self) -> nn.Module:
        return self._network

    def configure_optimizers(self):
        optim_dict = {}
        if self._optimizer == "AdamW":
            optim_dict["optimizer"] = torch.optim.AdamW(
                self._network.parameters(), lr=self._lr
            )
        elif self._optimizer == "Adam":
            optim_dict["optimizer"] = torch.optim.Adam(
                self._network.parameters(), lr=self._lr
            )
        else:
            raise ValueError(f"Unknown optimizer: {self._optimizer}")

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
        self._train_avg_loss(loss)
        self.log(
            "repr_train_avg_loss", self._train_avg_loss, on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        outputs, loss = self._step(batch)
        self._val_avg_loss(loss)
        self.log("repr_val_avg_loss", self._val_avg_loss, on_epoch=True, prog_bar=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._network(inputs)


@register_model("signal_model")
class SignalModel(L.LightningModule):
    def __init__(
        self,
        network: DictConfig,
        loss_func: DictConfig,
        lr: float,
        optimizer: str,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self._network = build_network(network)
        self._loss_func = build_loss_func(loss_func)
        self._lr = lr

        self._train_avg_loss = tm.MeanMetric()
        self._val_avg_loss = tm.MeanMetric()

        self._optimizer = optimizer

        self._repr_model: ReprModel | None = None

        if self._use_ASAM():
            self.automatic_optimization = False

    def set_repr_model(self, repr_model: ReprModel) -> None:
        self._repr_model = repr_model
        self._repr_model.eval()
        self._repr_model.requires_grad_(False)

    def _use_ASAM(self) -> bool:
        return self._optimizer == "ASAM"

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
        elif self._optimizer == "Adam":
            optim_dict["optimizer"] = torch.optim.Adam(
                self._network.parameters(), lr=self._lr
            )
        else:
            raise ValueError(f"Unknown optimizer: {self._optimizer}")

        return optim_dict

    def _compute_loss(
        self, outputs: dict[DFKey, torch.Tensor], labels: dict[DFKey, torch.Tensor]
    ) -> torch.Tensor:
        loss = self._loss_func(outputs, labels)
        return loss

    def _step(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        inputs, labels, _ = batch

        if self._repr_model is not None:
            inputs = self._repr_model.forward(inputs)

        outputs = self._network(inputs)
        loss = self._compute_loss(outputs, labels)
        return outputs, loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        outputs, loss = self._step(batch)

        if self._use_ASAM():
            opt = self.optimizers()
            self.manual_backward(loss)
            with torch.no_grad():
                opt.first_step(zero_grad=True)
                with torch.set_grad_enabled(True):
                    outputs_perturbed, loss_perturbed, _ = self._step(batch)
                    self.manual_backward(loss_perturbed)
                opt.second_step()
            opt.zero_grad()

        self._train_avg_loss(loss)
        self.trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.total.completed += (
            1
        )
        self.log(
            "signal_train_avg_loss", self._train_avg_loss, on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        outputs, loss = self._step(batch)
        self._val_avg_loss(loss)
        self.log(
            "signal_val_avg_loss", self._val_avg_loss, on_epoch=True, prog_bar=True
        )
        timestamp = batch[-1]
        self._store_outputs_to_df(self.trainer.datamodule.df, outputs, timestamp)

    def _store_outputs_to_df(
        self,
        df: pd.DataFrame,
        outputs: torch.Tensor,
        timeindex: torch.Tensor,
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


@register_model("signal_mixup")
class SignalMixupModel(SignalModel):
    def __init__(
        self,
        network: DictConfig,
        loss_func: DictConfig,
        lr: float,
        optimizer: str,
        mixup_alpha: float,
    ) -> None:
        super().__init__(network, loss_func, lr, optimizer)
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

    def _compute_loss_mixup(
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

    def _step(self, batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        inputs, labels, _ = batch

        if self._repr_model is not None:
            repr_outputs = self._repr_model.forward(inputs)

            if self.training:
                inputs_mixup, labels_a, labels_b, lambda_ = self._mixup(
                    repr_outputs, labels
                )
                outputs = self._network(inputs_mixup)
                loss = self._compute_loss_mixup(outputs, labels_a, labels_b, lambda_)
            else:
                outputs = self._network(repr_outputs)
                loss = self._compute_loss(outputs, labels)

            return outputs, loss, repr_outputs

        else:
            if self.training:
                inputs_mixup, labels_a, labels_b, lambda_ = self._mixup(inputs, labels)
                outputs = self._network(inputs_mixup)
                loss = self._compute_loss_mixup(outputs, labels_a, labels_b, lambda_)
            else:
                outputs = self._network(inputs)
                loss = self._compute_loss(outputs, labels)

            return outputs, loss, None
