import numpy as np
import torch
import torch.nn as nn

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


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-15, affine: float = True) -> None:
        super().__init__()
        self._eps = eps
        self._affine = affine
        self._num_features = num_features

        if self._affine:
            self._gamma = nn.Parameter(torch.ones(num_features))
            self._beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            self._compute_stats(x)
            x = self._normalize(x)
            if self._affine:
                x = self._transform(x)
            return x
        elif mode == "rev":
            if self._affine:
                x = self._retransform(x)
            x = self._denormalize(x)
            return x
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _compute_stats(self, x: torch.Tensor) -> None:
        self._mean = x.mean(dim=1, keepdim=True)
        self._var = x.var(dim=1, keepdim=True, unbiased=False)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._mean) / torch.sqrt(self._var + self._eps)

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sqrt(self._var + self._eps) + self._mean

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        return x * self._gamma[None, None, :] + self._beta[None, None, :]

    def _retransform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._beta[None, None, :]) / (self._gamma[None, None, :] + self._eps)


class LinearBatchRelu(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, lookback_num: int) -> None:
        super().__init__()
        self._in_ch = in_ch
        self._out_ch = out_ch

        self._linear = nn.Linear(in_ch, out_ch)
        self._norm = nn.InstanceNorm1d(lookback_num, affine=False)
        self._relu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._linear(x)
        x = self._norm(x)
        x = self._relu(x)
        return x


@register_network("nonlinear")
class Nonlinear(nn.Module):
    def __init__(
        self,
        input_ch: int,
        hidden_dim: int,
        lookback_num: int,
        num_outputs: int,
        moving_avg_kernel_size: int,
        use_repr: bool,
    ) -> None:
        super().__init__()

        self._input_ch = input_ch + 384 if use_repr else input_ch
        self._hidden_dim = hidden_dim
        self._lookback_num = lookback_num
        self._num_outputs = num_outputs

        self._input_norm = nn.InstanceNorm1d(lookback_num, affine=False)
        self._decompose = SeriesDecomp(moving_avg_kernel_size)
        self._seasonal_layer = LinearBatchRelu(input_ch, hidden_dim, lookback_num)
        self._trend_layer = LinearBatchRelu(input_ch, hidden_dim, lookback_num)
        self._encode_layer_1 = LinearBatchRelu(hidden_dim, hidden_dim, lookback_num)
        self._encode_layer_2 = LinearBatchRelu(hidden_dim, hidden_dim, lookback_num)
        self._encode_layer_3 = LinearBatchRelu(hidden_dim, hidden_dim, lookback_num)

        self._dropout = nn.Dropout()
        self._pred_linear = nn.Linear(self._lookback_num * hidden_dim, num_outputs)

    def _to_pred(self, output: torch.Tensor) -> tuple[dict[DFKey, torch.Tensor], dict[DFKey, torch.Tensor]]:
        output = output.view(output.shape[0], -1)
        output = self._dropout(output)
        pred = self._pred_linear(output).softmax(dim=-1)
        return pred

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs - inputs[:, 0:1]
        # inputs = self._input_norm(inputs)
        seasonal_x, trend_x = self._decompose(inputs)
        seasonal_x, trend_x = seasonal_x.permute(0, 2, 1), trend_x.permute(0, 2, 1)
        seasonal_x, trend_x = self._dropout(seasonal_x), self._dropout(trend_x)
        seasonal_output = self._seasonal_layer(seasonal_x)
        trend_output = self._trend_layer(trend_x)
        output = seasonal_output + trend_output
        output = self._encode_layer_1(output) + output
        output = self._encode_layer_2(output) + output
        output = self._encode_layer_3(output) + output
        return output.contiguous()

    def forward(self, inputs: torch.Tensor) -> tuple[dict[DFKey, torch.Tensor], dict[DFKey, torch.Tensor]]:
        output = self.encode(inputs)
        pred = self._to_pred(output).squeeze()
        return pred


@register_network("nonlinear_lookback")
class NonlinearLookback(nn.Module):
    def __init__(
        self,
        input_ch: int,
        hidden_dim: int,
        lookback_num: int,
        num_outputs: int,
        moving_avg_kernel_size: int,
        use_repr: bool,
    ) -> None:
        super().__init__()

        self._input_ch = input_ch + 384 if use_repr else input_ch
        self._hidden_dim = hidden_dim
        self._lookback_num = lookback_num
        self._num_outputs = num_outputs

        self._decompose = SeriesDecomp(moving_avg_kernel_size)
        self._seasonal_layer = LinearBatchRelu(input_ch, hidden_dim, lookback_num)
        self._trend_layer = LinearBatchRelu(input_ch, hidden_dim, lookback_num)

        self._encode_layer_1 = LinearBatchRelu(hidden_dim, hidden_dim, lookback_num)
        self._lookback_layer_1 = LinearBatchRelu(lookback_num, lookback_num // 2, hidden_dim)
        self._encode_layer_2 = LinearBatchRelu(hidden_dim, hidden_dim, lookback_num // 2)
        self._lookback_layer_2 = LinearBatchRelu(lookback_num // 2, lookback_num // 4, hidden_dim)
        self._encode_layer_3 = LinearBatchRelu(hidden_dim, hidden_dim, lookback_num // 4)
        self._lookback_layer_3 = LinearBatchRelu(lookback_num // 4, lookback_num // 8, hidden_dim)
        self._encode_layer_4 = LinearBatchRelu(hidden_dim, hidden_dim, lookback_num // 8)
        self._lookback_layer_4 = LinearBatchRelu(lookback_num // 8, lookback_num // 16, hidden_dim)

        self._dropout = nn.Dropout()
        self._pred_linear = nn.Linear(self._lookback_num // 16 * hidden_dim, num_outputs)

    def _to_pred(self, output: torch.Tensor) -> tuple[dict[DFKey, torch.Tensor], dict[DFKey, torch.Tensor]]:
        output = output.view(output.shape[0], -1)
        output = self._dropout(output)
        pred = self._pred_linear(output).softmax(dim=-1)
        return pred

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs - inputs[:, 0:1]

        seasonal_x, trend_x = self._decompose(inputs)
        seasonal_x, trend_x = seasonal_x.permute(0, 2, 1), trend_x.permute(0, 2, 1)
        seasonal_x, trend_x = self._dropout(seasonal_x), self._dropout(trend_x)
        seasonal_output = self._seasonal_layer(seasonal_x)
        trend_output = self._trend_layer(trend_x)
        output = seasonal_output + trend_output

        output = self._encode_layer_1(output) + output
        output = self._lookback_layer_1(output.permute(0, 2, 1)).permute(0, 2, 1)
        output = self._encode_layer_2(output) + output
        output = self._lookback_layer_2(output.permute(0, 2, 1)).permute(0, 2, 1)
        output = self._encode_layer_3(output) + output
        output = self._lookback_layer_3(output.permute(0, 2, 1)).permute(0, 2, 1)
        output = self._encode_layer_4(output) + output
        output = self._lookback_layer_4(output.permute(0, 2, 1)).permute(0, 2, 1)
        return output.contiguous()

    def forward(self, inputs: torch.Tensor) -> tuple[dict[DFKey, torch.Tensor], dict[DFKey, torch.Tensor]]:
        output = self.encode(inputs)
        pred = self._to_pred(output).squeeze()
        return pred


@register_network("nonlinear_maxpool")
class NonlinearMaxpool(Nonlinear):
    def __init__(
        self,
        input_ch: int,
        hidden_dim: int,
        lookback_num: int,
        num_outputs: int,
        moving_avg_kernel_size: int,
        use_repr: bool,
    ) -> None:
        super().__init__(input_ch, hidden_dim, lookback_num, num_outputs, moving_avg_kernel_size, use_repr)

        self._input_ch = input_ch
        self._hidden_dim = hidden_dim
        self._lookback_num = lookback_num
        self._num_outputs = num_outputs

        self._decompose = SeriesDecomp(moving_avg_kernel_size)
        self._seasonal_layer = LinearBatchRelu(input_ch, hidden_dim, lookback_num)
        self._trend_layer = LinearBatchRelu(input_ch, hidden_dim, lookback_num)
        self._encode_layer_1 = LinearBatchRelu(hidden_dim, hidden_dim, lookback_num)
        self._encode_layer_2 = LinearBatchRelu(hidden_dim, hidden_dim, lookback_num)
        self._encode_layer_3 = LinearBatchRelu(hidden_dim, hidden_dim, lookback_num)

        self._dropout = nn.Dropout()
        self._pred_linear = LinearBatchRelu(hidden_dim, num_outputs, 1)

    def _to_pred(self, output: torch.Tensor) -> tuple[dict[DFKey, torch.Tensor], dict[DFKey, torch.Tensor]]:
        # output, _ = output.max(1, keepdim=True)
        output = output.mean(1, keepdim=True)
        # output = self._dropout(output)
        pred = self._pred_linear(output).softmax(dim=-1)
        return pred


@register_network("mlpdropout")
class MLPDropout(nn.Module):
    def __init__(self, input_ch, hidden_dim, num_outputs, lookback_num, drop_prob, use_repr, *args, **kwargs):
        super().__init__()
        self._input_ch = input_ch + 384 if use_repr else input_ch
        self._num_outputs = num_outputs
        self._drop_prob = drop_prob
        init_ = lambda m: self.init(
            m,
            torch.nn.init.orthogonal_,
            lambda x: torch.nn.init.constant_(x, 0),
            np.sqrt(2),
        )
        self.linear = init_(torch.nn.Linear(self._input_ch * lookback_num, self._num_outputs))
        self.dropout = torch.nn.Dropout(p=self._drop_prob)

    def init(self, module, weight_init, bias_init, gain=1):
        weight_init(module.weight.data, gain=gain)
        bias_init(module.bias.data)
        return module

    def forward(self, inputs):
        x = self.dropout(inputs.view(inputs.shape[0], -1))
        x = self.linear(x).squeeze()
        x = x.softmax(dim=-1)
        return x
