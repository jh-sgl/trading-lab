import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics as tm
from omegaconf import DictConfig

from ...core.model.optim import SAM
from ...util.const import DFKey
from ...util.registry import build_loss_func, build_network, register_model
from .network.LaSTV4 import LaSTV4
from .network.repr_tta_dlinear import ReprTTADLinear
from .network.signal_causal_aware_classifier import SignalCausalAwareClassifier
from .network.signal_mlp_vq import SignalMLPVQ


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

    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        return self._network(inputs, **kwargs)


@register_model("repr_tta_model")
class ReprTTAModel(L.LightningModule):
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
        self._optimizer = optimizer

        self._train_avg_loss = tm.MeanMetric()
        self._val_avg_loss = tm.MeanMetric()

    @property
    def network(self) -> nn.Module:
        return self._network

    def configure_optimizers(self):
        if self._optimizer == "AdamW":
            optimizer = torch.optim.AdamW(self._network.parameters(), lr=self._lr)
        elif self._optimizer == "Adam":
            optimizer = torch.optim.Adam(self._network.parameters(), lr=self._lr)
        else:
            raise ValueError(f"Unknown optimizer: {self._optimizer}")

        return {"optimizer": optimizer}

    def _compute_loss(
        self,
        recon_z: torch.Tensor,
        pred_z: torch.Tensor,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        z1: torch.Tensor | None = None,
        z2: torch.Tensor | None = None,
        recon_z1: torch.Tensor | None = None,
        pred_z1: torch.Tensor | None = None,
        recon_z2: torch.Tensor | None = None,
        pred_z2: torch.Tensor | None = None,
    ) -> torch.Tensor:

        loss = self._loss_func(recon_z, inputs)
        loss = loss + self._loss_func(pred_z, labels)

        if z1 is not None and z2 is not None:
            z1 = F.normalize(z1, dim=-1)
            z2 = F.normalize(z2, dim=-1)

            temperature = 0.1
            sim_matrix = z1 @ z2.T  # (B, B)
            sim_matrix /= temperature
            sim_labels = torch.arange(z1.shape[0], device=z1.device)

            loss = loss + F.cross_entropy(sim_matrix, sim_labels)

        if all([item is not None for item in [recon_z1, pred_z1, recon_z2, pred_z2]]):
            loss = loss + self._loss_func(recon_z1, inputs)
            loss = loss + self._loss_func(pred_z1, labels)
            loss = loss + self._loss_func(recon_z2, inputs)
            loss = loss + self._loss_func(pred_z2, labels)

        return loss

    def _step(self, batch) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        inputs, labels, _ = batch
        if self.training:
            recon_z, pred_z, z1, z2, recon_z1, pred_z1, recon_z2, pred_z2 = (
                self._network(inputs, augment_z=True)
            )
            loss = self._compute_loss(
                recon_z,
                pred_z,
                inputs,
                labels,
                z1,
                z2,
                recon_z1,
                pred_z1,
                recon_z2,
                pred_z2,
            )
        else:
            recon_z, pred_z = self._network(inputs, augment_z=False)
            loss = self._compute_loss(recon_z, pred_z, inputs, labels)
        return (recon_z, pred_z), loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        (_, _), loss = self._step(batch)
        self._train_avg_loss(loss)
        self.log(
            "repr_train_avg_loss", self._train_avg_loss, on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        (_, _), loss = self._step(batch)
        self._val_avg_loss(loss)
        self.log("repr_val_avg_loss", self._val_avg_loss, on_epoch=True, prog_bar=True)

    def forward(
        self, inputs: torch.Tensor, tta_steps: int, tta_lr: float
    ) -> torch.Tensor:
        if tta_steps == 0:
            return self._network(inputs)[1]

        # --- Test-Time Adaptation ---
        inputs = inputs.detach()
        z_init = self.network.z_encoder(inputs.permute(0, 2, 1)).detach()
        z = torch.nn.Parameter(z_init, requires_grad=True)
        optimizer = torch.optim.Adam([z], lr=tta_lr)

        with torch.enable_grad():
            for _ in range(tta_steps):
                modulated = self.network(inputs, external_z=z)
                recon = modulated[0]  # recon from network
                recon_loss = nn.functional.mse_loss(recon, inputs)
                optimizer.zero_grad()
                recon_loss.backward()
                optimizer.step()

        # Final forward pass with optimized z
        _, pred = self.network(inputs, external_z=z.detach())
        return pred


@register_model("regime_model")
class RegimeModel(L.LightningModule):
    def __init__(
        self,
        network: DictConfig,
        loss_func: DictConfig,
        lr: float,
        optimizer: str,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self._network = build_network(network)
        self._loss_func = build_loss_func(loss_func)
        self._lr = lr

        self._train_avg_loss = tm.MeanMetric()
        self._val_avg_loss = tm.MeanMetric()

        self._train_acc = tm.Accuracy("multiclass", num_classes=num_classes)
        self._val_acc = tm.Accuracy("multiclass", num_classes=num_classes)

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
        repr_inputs, col_inputs, meta_inputs, labels, _ = batch
        outputs = self._network(repr_inputs, col_inputs, meta_inputs)
        loss = self._compute_loss(outputs, labels)
        return outputs, loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        outputs, loss = self._step(batch)
        self._train_avg_loss(loss)

        labels = batch[-2]
        self._train_acc(outputs, labels)
        self.log(
            "regime_train_avg_loss", self._train_avg_loss, on_epoch=True, prog_bar=True
        )
        self.log("regime_train_acc", self._train_acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        outputs, loss = self._step(batch)
        self._val_avg_loss(loss)

        labels = batch[-2]
        self._val_acc(outputs, labels)
        self.log(
            "regime_val_avg_loss", self._val_avg_loss, on_epoch=True, prog_bar=True
        )
        self.log("regime_valid_acc", self._val_acc, on_epoch=True, prog_bar=True)

    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        return self._network(inputs, **kwargs)


class Mixup:
    def __init__(self, alpha: float):
        self._alpha = alpha

    def __call__(
        self, inputs: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        if self._alpha <= 0.0:
            raise ValueError("alpha must be positive")

        lambda_ = np.random.beta(self._alpha, self._alpha)
        index = torch.randperm(inputs.size()[0]).to(inputs.device)
        inputs_mixup = lambda_ * inputs + (1 - lambda_) * inputs[index]
        labels_a, labels_b = labels, labels[index]
        return inputs_mixup, labels_a, labels_b, lambda_


@register_model("signal_model")
class SignalModel(L.LightningModule):
    def __init__(
        self,
        network: DictConfig,
        loss_func: DictConfig,
        lr: float,
        optimizer: str,
        mixup_alpha: float = 0.0,
        tta_steps: int = 10,
        tta_lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self._network = build_network(network)
        self._loss_func = build_loss_func(loss_func)
        self._lr = lr
        self._optimizer = optimizer

        self._train_avg_loss = tm.MeanMetric()
        self._val_avg_loss = tm.MeanMetric()

        self._mixup = Mixup(mixup_alpha) if mixup_alpha > 0.0 else None
        self._repr_model: ReprModel | None = None

        self._tta_steps = tta_steps
        self._tta_lr = tta_lr

        if self._use_ASAM():
            self.automatic_optimization = False

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
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        *,
        mixup: bool = False,
        labels_b: dict[DFKey, torch.Tensor] | None = None,
        lambda_: float = 1.0,
    ) -> torch.Tensor:
        loss_a = self._loss_func(outputs, labels)

        if mixup and labels_b is not None:
            loss_b = self._loss_func(outputs, labels_b)
            loss = lambda_ * loss_a + (1 - lambda_) * loss_b
        else:
            loss = loss_a

        return loss

    def _step(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        inputs, labels, _ = batch

        if isinstance(self._network, SignalMLPVQ):
            outputs, vq_loss, _ = self._network(inputs)
            loss = self._compute_loss(outputs, labels) + vq_loss

        elif self.training:
            if isinstance(
                self._network, SignalCausalAwareClassifier
            ):  # turn off mixup for causality
                outputs, causal_consistency_loss = self._network(
                    inputs, compute_consistency=True
                )
                loss = self._compute_loss(outputs, labels) + causal_consistency_loss
            elif self._mixup is not None:
                inputs_mixup, labels_a, labels_b, lambda_ = self._mixup(inputs, labels)
                outputs = self._network(inputs_mixup)
                loss = self._compute_loss(
                    outputs, labels_a, mixup=True, labels_b=labels_b, lambda_=lambda_
                )
            else:
                outputs = self._network(inputs)
                loss = self._compute_loss(outputs, labels)

        else:
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
                    outputs_perturbed, loss_perturbed = self._step(batch)
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
