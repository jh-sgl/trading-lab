import numpy as np
import torch
import torch.nn as nn

from ...util.registry import register_network


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


@register_network("mlp_dropout")
class MLPDropout(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout_rate: float,
        lookback_num: int,
    ) -> None:
        super().__init__()

        init_ = lambda m: self._init(
            m,
            torch.nn.init.orthogonal_,
            lambda x: torch.nn.init.constant_(x, 0),
            np.sqrt(2),
        )

        self.linear = init_(torch.nn.Linear(input_dim * lookback_num, output_dim))
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def _init(self, module, weight_init, bias_init, gain=1):
        weight_init(module.weight.data, gain=gain)
        bias_init(module.bias.data)
        return module

    def forward(self, inputs):
        bs = inputs.shape[0]
        inputs = inputs.view(bs, -1)
        x = self.dropout(inputs)
        return self.linear(x).softmax(dim=-1)


@register_network("dlinear")
class DLinear(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(
        self,
        input_ch: int,
        hidden_num: int,
        output_num: int,
        lookback_num: int,
        moving_avg_kernel_size: int,
    ) -> None:
        super(DLinear, self).__init__()

        self._decompose = SeriesDecomp(moving_avg_kernel_size)
        self._linear_seasonal_enc = nn.Linear(input_ch, hidden_num)
        self._linear_trend_enc = nn.Linear(input_ch, hidden_num)

        self._linear_seasonal_enc.weight = nn.Parameter(
            (1 / input_ch) * torch.ones([hidden_num, input_ch])
        )
        self._linear_trend_enc.weight = nn.Parameter(
            (1 / input_ch) * torch.ones([hidden_num, input_ch])
        )
        self._dropout = nn.Dropout()

        self._linear_predictor = nn.Linear(
            hidden_num * input_ch * lookback_num, output_num
        )

    def _to_pred(self, output: torch.Tensor) -> torch.Tensor:
        output = output.view(output.shape[0], -1)
        output = self._dropout(output)
        pred = self._linear_predictor(output).squeeze(-1)
        return pred

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        seasonal_x, trend_x = self._decompose(inputs)
        seasonal_output = self._linear_seasonal_enc(seasonal_x)
        trend_output = self._linear_trend_enc(trend_x)
        output = seasonal_output + trend_output
        pred = self._to_pred(output)
        return pred
