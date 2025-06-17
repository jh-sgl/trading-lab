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


@register_network("repr_dlinear")
class ReprDLinear(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(
        self,
        repr_lookback_num: int,
        repr_lookahead_num: int,
        moving_avg_kernel_size: int,
    ) -> None:
        super().__init__()

        self._decompose = SeriesDecomp(moving_avg_kernel_size)
        self._linear_seasonal_enc = nn.Linear(repr_lookback_num, repr_lookahead_num)
        self._linear_trend_enc = nn.Linear(repr_lookback_num, repr_lookahead_num)

        self._linear_seasonal_enc.weight = nn.Parameter(
            (1 / repr_lookback_num)
            * torch.ones([repr_lookahead_num, repr_lookback_num])
        )
        self._linear_trend_enc.weight = nn.Parameter(
            (1 / repr_lookback_num)
            * torch.ones([repr_lookahead_num, repr_lookback_num])
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        seasonal_x, trend_x = self._decompose(inputs)
        seasonal_output = self._linear_seasonal_enc(seasonal_x)
        trend_output = self._linear_trend_enc(trend_x)
        output = seasonal_output + trend_output
        return output.permute(0, 2, 1)


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


@register_network("repr_nonlinear")
class ReprNonLinear(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(
        self,
        input_dim: int,
        repr_lookback_num: int,
        repr_lookahead_num: int,
        moving_avg_kernel_size: int,
    ) -> None:
        super().__init__()

        self._decompose = SeriesDecomp(moving_avg_kernel_size)

        self._linear_seasonal_enc = InFcGelu(repr_lookback_num, repr_lookback_num)
        self._linear_trend_enc = InFcGelu(repr_lookback_num, repr_lookback_num)

        self._linear_lookback_enc_1 = InFcGelu(
            repr_lookback_num, repr_lookback_num // 2
        )
        self._linear_ch_enc_1 = InFcGelu(
            input_dim, input_dim, permute_input=True, permute_output=True
        )
        self._linear_lookback_enc_2 = InFcGelu(
            repr_lookback_num // 2, repr_lookback_num // 4
        )
        self._linear_ch_enc_2 = InFcGelu(
            input_dim, input_dim, permute_input=True, permute_output=True
        )
        self._linear_lookback_enc_3 = InFcGelu(
            repr_lookback_num // 4, repr_lookback_num // 8
        )
        self._linear_ch_enc_3 = InFcGelu(
            input_dim, input_dim, permute_input=True, permute_output=True
        )

        self._to_repr = nn.Linear(repr_lookback_num // 8, repr_lookahead_num)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        seasonal_x, trend_x = self._decompose(inputs)
        seasonal_output = self._linear_seasonal_enc(seasonal_x)
        trend_output = self._linear_trend_enc(trend_x)
        output = seasonal_output + trend_output

        output = self._linear_lookback_enc_1(output)
        output = self._linear_ch_enc_1(output) + output
        output = self._linear_lookback_enc_2(output)
        output = self._linear_ch_enc_2(output) + output
        output = self._linear_lookback_enc_3(output)
        output = self._linear_ch_enc_3(output) + output
        output = self._to_repr(output)

        return output.permute(0, 2, 1)


@register_network("signal_mlpdropout")
class SignalMLPDropout(nn.Module):
    def __init__(
        self,
        input_dim: int,
        repr_lookahead_num: int,
        output_dim: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()

        init_ = lambda m: self._init(
            m,
            torch.nn.init.orthogonal_,
            lambda x: torch.nn.init.constant_(x, 0),
            np.sqrt(2),
        )

        self.linear = init_(torch.nn.Linear(input_dim * repr_lookahead_num, output_dim))
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def _init(self, module, weight_init, bias_init, gain=1):
        weight_init(module.weight.data, gain=gain)
        bias_init(module.bias.data)
        return module

    def forward(self, inputs):
        bs = inputs.shape[0]
        inputs = inputs.reshape(bs, -1)
        x = self.dropout(inputs)
        return self.linear(x).softmax(dim=-1)
