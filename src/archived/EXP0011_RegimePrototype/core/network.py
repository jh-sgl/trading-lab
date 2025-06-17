import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from util.registry import register_network

from .const import DFKey


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


class LinearBatchRelu(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, lookback_num: int | None) -> None:
        super().__init__()
        self._in_ch = in_ch
        self._out_ch = out_ch

        self._linear = nn.Linear(in_ch, out_ch)
        self._norm = nn.InstanceNorm1d(lookback_num) if lookback_num is not None else F.instance_norm
        self._relu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._linear(x)
        x = self._norm(x)
        x = self._relu(x)
        return x


@register_network("basic")
class Basic(nn.Module):
    def __init__(
        self,
        input_ch: int,
        hidden_dim: int,
        lookback_num_longterm: int,
        lookback_num_shortterm: int,
        lookahead_num: int,
        regime_num: int,
        moving_avg_kernel_size: int,
    ) -> None:
        super().__init__()

        self._input_ch = input_ch + 1  # 1 for date token
        self._hidden_dim = hidden_dim
        self._lookback_num_longterm = lookback_num_longterm
        self._lookback_num_shortterm = lookback_num_shortterm

        self._decompose = SeriesDecomp(moving_avg_kernel_size)
        self._seasonal_layer = LinearBatchRelu(self._input_ch, hidden_dim, None)
        self._trend_layer = LinearBatchRelu(self._input_ch, hidden_dim, None)

        self._input_encoder = LinearBatchRelu(input_ch, hidden_dim, None)
        self._regime_encoder = RegimeEncoder(lookback_num_longterm, hidden_dim, regime_num)
        self._regime_prototypes = nn.Parameter(
            torch.randn(regime_num, self._lookback_num_shortterm, hidden_dim, lookahead_num, 3)
        )

        self._dropout = nn.Dropout()

    def _encode_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs - inputs[:, 0:1]
        seasonal_x, trend_x = self._decompose(inputs)
        seasonal_x, trend_x = seasonal_x.permute(0, 2, 1), trend_x.permute(0, 2, 1)
        seasonal_x, trend_x = self._dropout(seasonal_x), self._dropout(trend_x)
        seasonal_output = self._seasonal_layer(seasonal_x)
        trend_output = self._trend_layer(trend_x)
        encoded = seasonal_output + trend_output
        return encoded

    def forward(self, inputs_longterm: torch.Tensor, inputs_shortterm: torch.Tensor) -> torch.Tensor:
        encoded_longterm = self._encode_inputs(inputs_longterm)
        encoded_shortterm = self._encode_inputs(inputs_shortterm)

        regime_idx = self._regime_encoder(encoded_longterm)
        regime_weights = F.gumbel_softmax(regime_idx, tau=1.0, hard=True)
        weighted_prototypes = regime_weights[:, :, None, None, None, None] * self._regime_prototypes[None, ...]
        selected_prototypes = weighted_prototypes.sum(dim=1)

        output = torch.einsum("bnh, bnhmo -> bmo", encoded_shortterm, selected_prototypes)
        output = torch.softmax(output.view(output.shape[0], -1), dim=-1).view_as(output)
        return output


class RegimeEncoder(nn.Module):
    def __init__(self, lookback_num_longterm: int, hidden_dim: int, regime_num: int) -> None:
        super().__init__()
        self._hidden_dim_layer_1 = LinearBatchRelu(hidden_dim, hidden_dim, lookback_num_longterm)
        self._lookback_layer_1 = LinearBatchRelu(lookback_num_longterm, lookback_num_longterm // 2, hidden_dim)
        self._hidden_dim_layer_2 = LinearBatchRelu(hidden_dim, hidden_dim, lookback_num_longterm // 2)
        self._lookback_layer_2 = LinearBatchRelu(lookback_num_longterm // 2, lookback_num_longterm // 4, hidden_dim)
        self._hidden_dim_layer_3 = LinearBatchRelu(hidden_dim, hidden_dim, lookback_num_longterm // 4)
        self._lookback_layer_3 = LinearBatchRelu(lookback_num_longterm // 4, lookback_num_longterm // 8, hidden_dim)

        self._to_regime_idx = nn.Linear(hidden_dim * lookback_num_longterm // 8, regime_num)

    def forward(self, encoded_longterm: torch.Tensor) -> torch.Tensor:
        output = self._hidden_dim_layer_1(encoded_longterm)
        output = self._lookback_layer_1(output.permute(0, 2, 1)).permute(0, 2, 1)
        output = self._hidden_dim_layer_2(output)
        output = self._lookback_layer_2(output.permute(0, 2, 1)).permute(0, 2, 1)
        output = self._hidden_dim_layer_3(output)
        output = self._lookback_layer_3(output.permute(0, 2, 1)).permute(0, 2, 1)
        regime_idx = self._to_regime_idx(output.flatten(1))
        return regime_idx
