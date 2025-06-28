import torch
import torch.nn as nn
from omegaconf import DictConfig

from ...util.registry import build_loss_func, register_loss_func


@register_loss_func("focal")
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 1, gamma: float = 2, eps: float = 1e-10) -> None:
        super().__init__()
        self._alpha = alpha
        self._gamma = gamma
        self._eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        ce_loss = -(y_true * torch.log(y_pred + self._eps)).sum(dim=-1)
        pt = torch.exp(-ce_loss)
        focal_loss = self._alpha * (1 - pt) ** self._gamma * ce_loss
        return focal_loss.mean()


@register_loss_func("reversefocal")
class ReverseFocalLoss(nn.Module):
    def __init__(self, alpha: float = 1, gamma: float = 2, eps: float = 1e-10) -> None:
        super().__init__()
        self._alpha = alpha
        self._gamma = gamma
        self._eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        ce_loss = -(y_pred * torch.log(y_true + self._eps)).sum(dim=1)
        pt = torch.exp(-ce_loss)
        focal_loss = self._alpha * (1 - pt) ** self._gamma * ce_loss
        return focal_loss.mean()


@register_loss_func("mse")
class MSELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._mse_loss = nn.MSELoss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return self._mse_loss(y_pred, y_true)


@register_loss_func("l1")
class L1Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._l1_loss = nn.L1Loss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return self._l1_loss(y_pred, y_true)


@register_loss_func("pairwisemarginranking")
class PairwiseMarginRankingLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        self._margin_ranking_loss = nn.MarginRankingLoss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        loss = torch.tensor(0.0, device=y_pred.device)
        for i, j in self._pairs:
            pred_i, pred_j = y_pred[:, i], y_pred[:, j]
            true_i, true_j = y_true[:, i], y_true[:, j]

            sign = torch.sign(true_i - true_j)
            loss += self._margin_ranking_loss(pred_i, pred_j, sign)
        return loss / len(self._pairs)


@register_loss_func("distpred")
class DistPredLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred, label) -> torch.Tensor:
        return self._crps_ensemble(label, pred)

    def _crps_ensemble(
        self,
        observations,
        forecasts,
        axis: int = -1,
        sorted_ensemble: bool = False,
        estimator: str = "pwm",
    ):
        r"""Estimate the Continuous Ranked Probability Score (CRPS) for a finite ensemble.

        Parameters
        ----------
        observations: ArrayLike
            The observed values.
        forecasts: ArrayLike
            The predicted forecast ensemble, where the ensemble dimension is by default
            represented by the last axis.
        axis: int
            The axis corresponding to the ensemble. Default is the last axis.
        sorted_ensemble: bool
            Boolean indicating whether the ensemble members are already in ascending order.
            Default is False.
        estimator: str
            Indicates the CRPS estimator to be used.
        backend: str
            The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

        Returns
        -------
        crps: ArrayLike
            The CRPS between the forecast ensemble and obs.

        Examples
        --------
        >>> from scoringrules import crps
        >>> crps.ensemble(pred, obs)
        """

        if estimator not in ["nrg", "pwm", "fair"]:
            raise ValueError(f"{estimator} is not a valid estimator. ")

        if axis != -1:
            forecasts = torch.moveaxis(forecasts, axis, -1)

        if not sorted_ensemble and estimator not in ["nrg", "fair"]:
            forecasts, _ = torch.sort(forecasts, axis=-1)

        return self._ensemble(observations, forecasts, estimator)

    def _ensemble(self, obs, fct, estimator: str = "pwm"):
        """Compute the CRPS for a finite ensemble."""
        if estimator == "nrg":
            out = self._crps_ensemble_nrg(obs, fct)
        elif estimator == "pwm":
            out = self._crps_ensemble_pwm(obs, fct)
        elif estimator == "fair":
            out = self._crps_ensemble_fair(obs, fct)
        else:
            raise ValueError("no estimator specified for ensemble!")
        return out.mean()

    def _crps_ensemble_fair(self, obs, fct):
        """Fair version of the CRPS estimator based on the energy form."""
        M: int = fct.shape[-1]
        e_1 = torch.sum(torch.abs(obs[..., None] - fct), axis=-1) / M
        e_2 = torch.sum(
            torch.abs(fct[..., None] - fct[..., None, :]),
            axis=(-1, -2),
        ) / (M * (M - 1))
        return e_1 - 0.5 * e_2

    def _crps_ensemble_nrg(self, obs, fct):
        """CRPS estimator based on the energy form."""
        M: int = fct.shape[-1]
        e_1 = torch.sum(torch.abs(obs[..., None] - fct), axis=-1) / M
        e_2 = torch.sum(torch.abs(fct[..., None] - fct[..., None, :]), (-1, -2)) / (
            M**2
        )
        return e_1 - 0.5 * e_2

    def _crps_ensemble_pwm(self, obs, fct):
        """CRPS estimator based on the probability weighted moment (PWM) form."""
        M: int = fct.shape[-1]
        expected_diff = torch.sum(torch.abs(obs[..., None] - fct), axis=-1) / M
        β_0 = torch.sum(fct, axis=-1) / M
        β_1 = torch.sum(fct * torch.arange(M).to(fct.device), axis=-1) / (M * (M - 1.0))
        return expected_diff + β_0 - 2.0 * β_1


@register_loss_func("lastv4")
class LastV4Loss(nn.Module):
    def __init__(self, base_loss_func: DictConfig) -> None:
        super().__init__()
        self._base_loss_func = build_loss_func(base_loss_func)

    def forward(
        self,
        outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, None],
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        pred, elbo, mlbo, _ = outputs
        loss = self._base_loss_func(pred, y_true) - elbo - mlbo
        return loss
