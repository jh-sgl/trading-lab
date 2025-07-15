import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....util.registry import register_network


class MovingAvg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size: int, stride: int):
        super().__init__()
        self._kernel_size = kernel_size
        self._avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # padding on the both ends of time series
        front = x[:, :1, :].repeat(1, (self._kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self._kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)

        x = self._avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class SeriesDecomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size: int, stride: int = 1) -> None:
        super().__init__()
        self._moving_avg = MovingAvg(kernel_size, stride)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        moving_mean = self._moving_avg(x)
        residue = x - moving_mean
        return residue.permute(0, 2, 1), moving_mean.permute(0, 2, 1)


@register_network("repr_tta_dlinear")
class ReprTTADLinear(nn.Module):
    """
    Regime-aware DLinear with LoRA-style output modulation and test-time adaptation.
    Inspired by https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(
        self,
        repr_lookback_num: int,
        repr_lookahead_num: int,
        moving_avg_kernel_size: int,
        z_dim: int = 32,
        lora_rank: int = 4,
    ) -> None:
        super().__init__()

        self._decompose = SeriesDecomp(moving_avg_kernel_size)
        self._linear_seasonal_enc = nn.Linear(
            repr_lookback_num, repr_lookahead_num + repr_lookback_num
        )
        self._linear_trend_enc = nn.Linear(
            repr_lookback_num, repr_lookahead_num + repr_lookback_num
        )

        # Initialize weights with averaging scheme
        self._linear_seasonal_enc.weight = nn.Parameter(
            (1 / repr_lookback_num)
            * torch.ones([repr_lookahead_num + repr_lookback_num, repr_lookback_num])
        )
        self._linear_trend_enc.weight = nn.Parameter(
            (1 / repr_lookback_num)
            * torch.ones([repr_lookahead_num + repr_lookback_num, repr_lookback_num])
        )

        # Regime encoder (can be learned or optimized per sequence)
        self.z_encoder = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, z_dim),
        )

        # LoRA-style modulator
        self.A_proj = nn.Linear(z_dim, 16 * lora_rank)
        self.B_proj = nn.Linear(z_dim, lora_rank * 16)
        self.bias_proj = nn.Linear(z_dim, 16)
        self.lora_rank = lora_rank

        self._augmentor = HybridAugmentor()

    def forward(
        self,
        inputs: torch.Tensor,
        augment_z: bool = False,
        external_z: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # inputs: (B, T, C) where T = repr_lookback_num and C = 16
        seasonal_x, trend_x = self._decompose(inputs)
        seasonal_out = self._linear_seasonal_enc(seasonal_x)  # (B, C, T')
        trend_out = self._linear_trend_enc(trend_x)
        base_out = seasonal_out + trend_out  # (B, C, T')
        base_out = base_out.permute(0, 2, 1)  # (B, T', C)

        if external_z is not None:
            z = external_z
        else:
            z = self.z_encoder(inputs.permute(0, 2, 1))  # (B, z_dim)

        recon_z, pred_z = self._modulate(z, base_out, inputs)

        if augment_z:
            aug_input = self._augmentor.augment_input(inputs)
            z1 = self.z_encoder(aug_input.permute(0, 2, 1))
            z2 = self._augmentor.augment_latent(z)

            recon_z1, pred_z1 = self._modulate(z1, base_out, inputs)
            recon_z2, pred_z2 = self._modulate(z2, base_out, inputs)
            return recon_z, pred_z, z1, z2, recon_z1, pred_z1, recon_z2, pred_z2

        return recon_z, pred_z

    def _modulate(
        self, z: torch.Tensor, base_out: torch.Tensor, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Regime token z from input
        A = self.A_proj(z).view(-1, 16, self.lora_rank)  # (B, 16, r)
        B = self.B_proj(z).view(-1, self.lora_rank, 16)  # (B, r, 16)
        delta_W = torch.bmm(A, B)  # (B, 16, 16)
        bias = self.bias_proj(z).unsqueeze(1)  # (B, 1, 16)

        # Apply modulation: x + x @ W(z)^T + b(z)
        mod_out = base_out + torch.matmul(base_out, delta_W.transpose(1, 2)) + bias

        # Split output
        recon = mod_out[:, : inputs.size(1)]  # (B, 100, 16)
        pred = mod_out[:, inputs.size(1) :]  # (B, 24, 16)
        return recon, pred


class InFcGelu(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        permute_input: bool = False,
        permute_output: bool = False,
    ) -> None:
        super().__init__()
        self._fc = nn.Linear(input_dim, output_dim)
        self._in = nn.InstanceNorm1d(input_dim)
        self._gelu = nn.GELU()
        self._permute_input = permute_input
        self._permute_output = permute_output

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self._permute_input:
            inputs = inputs.permute(0, 2, 1)
        x = self._gelu(self._fc(self._in(inputs)))
        if self._permute_output:
            x = x.permute(0, 2, 1)
        return x


class HybridAugmentor:
    def __init__(
        self,
        dropout_p=0.4,
        noise_std_scale=0.5,
        time_warp_prob=0.3,
    ):
        self.dropout_p = dropout_p
        self.noise_std_scale = (
            noise_std_scale  # scale factor of stddev from original input/z stats
        )
        self.time_warp_prob = time_warp_prob

    def apply_dropout(self, x):
        return F.dropout(x, p=self.dropout_p, training=True)

    def apply_adaptive_noise(self, x):
        # x: (B, ..., D)
        std = torch.std(x, dim=1, keepdim=True)  # dimension-wise std
        noise = torch.randn_like(x) * std * self.noise_std_scale
        return x + noise

    def apply_time_warp(self, x):
        # x: (B, T, C), time-major input
        if self.time_warp_prob <= 0 or random.random() > self.time_warp_prob:
            return x

        B, T, C = x.shape
        warp_factor = random.uniform(0.8, 1.2)
        new_T = max(4, int(T * warp_factor))
        x_resampled = F.interpolate(
            x.permute(0, 2, 1), size=new_T, mode="linear", align_corners=False
        )
        x_resampled = F.interpolate(
            x_resampled, size=T, mode="linear", align_corners=False
        )
        return x_resampled.permute(0, 2, 1)

    def augment_input(self, inputs):
        x = self.apply_dropout(inputs)
        x = self.apply_adaptive_noise(x)
        x = self.apply_time_warp(x)
        return x

    def augment_latent(self, z):
        z = self.apply_dropout(z)
        z = self.apply_adaptive_noise(z)
        return z

    def hybrid_augment(self, inputs, z_encoder):
        """
        Apply both input and latent augmentation.

        Args:
            inputs: torch.Tensor of shape (B, T, C)
            z_encoder: encoder module that returns z from inputs

        Returns:
            Tuple of (z1, z2):
              - z1: z from input-augmented view
              - z2: z from latent-augmented view
        """
        return z1, z2
