import lightning as L
import numpy as np
import pandas as pd
import torch
import torchmetrics as tm
from omegaconf import DictConfig

from ...util.const import DFKey
from ...util.registry import build_loss_func, build_network, register_model
from ..model import ReprModel
from .network.LaSTV4 import LaSTV4
from .network.repr_tta_dlinear import ReprTTADLinear
from .optim import SAM


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


@register_model("signal_vq_model")
class SignalVQModel(L.LightningModule):
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
        pred: torch.Tensor,
        labels: torch.Tensor,
        pred_delta: torch.Tensor,
        retrieved_label_mean: torch.Tensor,
        *,
        mixup: bool = False,
        labels_b: torch.Tensor | None = None,
        lambda_: float = 1.0,
    ) -> torch.Tensor:
        loss_a = self._loss_func(pred, labels)
        a_delta_gt = labels - retrieved_label_mean
        loss_a_delta = self._loss_func(pred_delta, a_delta_gt)
        loss_a = loss_a + loss_a_delta

        if mixup and labels_b is not None:
            loss_b = self._loss_func(pred, labels_b)
            b_delta_gt = labels_b - retrieved_label_mean
            loss_b_delta = self._loss_func(pred_delta, b_delta_gt)
            loss_b = loss_b + loss_b_delta
            loss = lambda_ * loss_a + (1 - lambda_) * loss_b
        else:
            loss = loss_a

        return loss

    def _step(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        inputs, labels, _ = batch

        if self.training and self._mixup is not None:
            inputs_mixup, labels_a, labels_b, lambda_ = self._mixup(inputs, labels)
            pred, pred_delta, retrieved_labels_mean = self._network(inputs_mixup)
            loss = self._compute_loss(
                pred,
                labels_a,
                pred_delta,
                retrieved_labels_mean,
                mixup=True,
                labels_b=labels_b,
                lambda_=lambda_,
            )
        else:
            pred, pred_delta, retrieved_labels_mean = self._network(inputs)
            loss = self._compute_loss(pred, labels, pred_delta, retrieved_labels_mean)

        return pred, loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        output, loss = self._step(batch)

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

    def on_train_epoch_start(self) -> None:
        """Build causal historical index for training."""
        inputs = self.trainer.datamodule.signal_train_repr_tensor
        labels = self.trainer.datamodule.signal_train_label_tensor
        self._network.build_historical_index_for_train(inputs, labels)

    def on_validation_epoch_start(self) -> None:
        """Build global historical index for validation."""
        inputs = self.trainer.datamodule.signal_train_repr_tensor
        labels = self.trainer.datamodule.signal_train_label_tensor
        self._network.build_historical_index_all(inputs, labels)
