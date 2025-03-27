import torch
import torch.nn as nn

from util.registry import register_loss_func


@register_loss_func("Focal")
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 1, gamma: float = 2, eps: float = 1e-10) -> None:
        super().__init__()
        self._alpha = alpha
        self._gamma = gamma
        self._eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        ce_loss = -(y_true * torch.log(y_pred + self._eps)).sum(dim=1)
        pt = torch.exp(-ce_loss)
        focal_loss = self._alpha * (1 - pt) ** self._gamma * ce_loss
        return focal_loss.mean()
