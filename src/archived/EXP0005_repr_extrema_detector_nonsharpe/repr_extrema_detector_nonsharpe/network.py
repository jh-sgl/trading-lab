import numpy as np
import torch
import torch.nn as nn

from util.registry import register_network


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


@register_network("ReprExtremaDetectorNonsharpeV1")
class ReprExtremaDetectorNonsharpeV1(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(
        self,
        input_ch: int,
        hidden_dim: int,
        output_num: int,
        input_lookback_num: int,
        moving_avg_kernel_size: int,
    ) -> None:
        super().__init__()
        self._decompose = SeriesDecomp(moving_avg_kernel_size)

        self._linear_seasonal_enc = nn.Linear(input_lookback_num, hidden_dim)
        self._linear_trend_enc = nn.Linear(input_lookback_num, hidden_dim)

        self._linear_seasonal_enc.weight = nn.Parameter(
            (1 / input_lookback_num) * torch.ones([hidden_dim, input_lookback_num])
        )
        self._linear_trend_enc.weight = nn.Parameter(
            (1 / input_lookback_num) * torch.ones([hidden_dim, input_lookback_num])
        )
        self._dropout = nn.Dropout()
        self._linear_predictor = nn.Linear(hidden_dim * input_ch, output_num)

    def _to_pred(self, output: torch.Tensor) -> torch.Tensor:
        output = output.view(output.shape[0], -1)
        output = self._dropout(output)
        pred = self._linear_predictor(output).squeeze(-1)
        return pred

    def forward(self, candle_inputs: torch.Tensor) -> torch.Tensor:
        seasonal_x, trend_x = self._decompose(candle_inputs)

        seasonal_output = self._linear_seasonal_enc(seasonal_x)
        trend_output = self._linear_trend_enc(trend_x)
        output = seasonal_output + trend_output
        pred = self._to_pred(output)

        return pred


class LinearBatchRelu(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self._in_ch = in_ch
        self._out_ch = out_ch

        self._linear = nn.Linear(in_ch, out_ch)
        self._instancenorm = nn.InstanceNorm1d(out_ch)
        self._relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self._linear(x)
        x = x.permute(0, 2, 1)
        x = self._instancenorm(x)
        x = self._relu(x)
        return x


@register_network("ReprExtremaDetectorNonsharpeV2")
class ReprExtremaDetectorNonsharpeV2(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(
        self,
        input_ch: int,
        hidden_dim: int,
        output_num: int,
        input_lookback_num: int,
        moving_avg_kernel_size: int,
    ) -> None:
        super().__init__()
        self._decompose = SeriesDecomp(moving_avg_kernel_size)

        self._seasonal_layer = LinearBatchRelu(input_ch, hidden_dim)
        self._trend_layer = LinearBatchRelu(input_ch, hidden_dim)

        self._dropout = nn.Dropout()
        self._relu_predictor = nn.LeakyReLU(0.1, inplace=True)
        self._linear_predictor = nn.Linear(input_lookback_num * hidden_dim, output_num)

    def _to_pred(self, output: torch.Tensor) -> torch.Tensor:
        output = output.view(output.shape[0], -1)
        output = self._relu_predictor(output)
        pred = self._linear_predictor(output).squeeze(-1)
        return pred

    def forward(self, candle_inputs: torch.Tensor) -> torch.Tensor:
        seasonal_x, trend_x = self._decompose(candle_inputs)
        seasonal_x = self._seasonal_layer(seasonal_x)
        trend_x = self._trend_layer(trend_x)

        output = seasonal_x + trend_x
        pred = self._to_pred(output)

        return pred


@register_network("ReprExtremaDetectorNonsharpeMLPDropout")
class ReprExtremaDetectorNonsharpeMLPDropout(nn.Module):
    def __init__(self, num_inputs, num_outputs=3, drop_prob=0.3, *args, **kwargs):
        super().__init__()

        init_ = lambda m: self.init(
            m,
            torch.nn.init.orthogonal_,
            lambda x: torch.nn.init.constant_(x, 0),
            np.sqrt(2),
        )

        self.linear = init_(torch.nn.Linear(num_inputs, num_outputs))
        self.dropout = torch.nn.Dropout(p=drop_prob)

    def init(self, module, weight_init, bias_init, gain=1):
        weight_init(module.weight.data, gain=gain)
        bias_init(module.bias.data)
        return module

    def forward(self, inputs):
        bs = inputs.shape[0]
        inputs = inputs.view(bs, -1)
        x = self.dropout(inputs)
        return self.linear(x)
