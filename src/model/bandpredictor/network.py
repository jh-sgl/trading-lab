import torch
import torch.nn as nn

from model.bandpredictor.dataset import PastInputs


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


class DLinear(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        input_ch: int,
        output_num: int,
        lookback_days: int,
        moving_avg_kernel_size: int,
    ) -> None:
        super(DLinear, self).__init__()
        self._seq_len = seq_len
        self._pred_len = pred_len
        self._decompose = SeriesDecomp(moving_avg_kernel_size)

        self._instancenorm_candle_inputs = nn.InstanceNorm1d(input_ch)
        self._linear_seasonal_enc = nn.Linear(self._seq_len, self._pred_len)
        self._linear_trend_enc = nn.Linear(self._seq_len, self._pred_len)
        self._instancenorm_candle_output = nn.InstanceNorm1d(self._pred_len)

        self._linear_seasonal_enc.weight = nn.Parameter(
            (1 / self._seq_len) * torch.ones([self._pred_len, self._seq_len])
        )
        self._linear_trend_enc.weight = nn.Parameter((1 / self._seq_len) * torch.ones([self._pred_len, self._seq_len]))
        self._dropout = nn.Dropout()
        self._linear_predictor = nn.Linear(self._pred_len * input_ch + 3 * lookback_days, output_num)

    def _to_pred(self, output: torch.Tensor) -> torch.Tensor:
        output = output.view(output.shape[0], -1)
        output = self._dropout(output)
        pred = self._linear_predictor(output).squeeze(-1)
        return pred

    def forward(self, past_inputs: PastInputs) -> torch.Tensor:
        candle_inputs = past_inputs.candle_inputs
        past_band_center = past_inputs.past_band_center
        past_band_upperbound = past_inputs.past_band_upperbound
        past_band_lowerbound = past_inputs.past_band_lowerbound

        candle_inputs = self._instancenorm_candle_inputs(candle_inputs.permute(0, 2, 1)).permute(0, 2, 1)
        seasonal_x, trend_x = self._decompose(candle_inputs)

        seasonal_output = self._linear_seasonal_enc(seasonal_x)
        trend_output = self._linear_trend_enc(trend_x)
        output = seasonal_output + trend_output
        output = self._instancenorm_candle_output(output.permute(0, 2, 1)).permute(0, 2, 1).contiguous()

        past_band_center = self._dropout(past_band_center)
        past_band_upperbound = self._dropout(past_band_upperbound)
        past_band_lowerbound = self._dropout(past_band_lowerbound)

        pred = self._to_pred(
            torch.cat(
                [output.view(output.shape[0], -1), past_band_center, past_band_upperbound, past_band_lowerbound], dim=-1
            )
        )
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


class DLinearV2(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        input_ch: int,
        output_num: int,
        lookback_days: int,
        moving_avg_kernel_size: int,
    ) -> None:
        super().__init__()
        self._seq_len = seq_len
        self._pred_len = pred_len
        self._decompose = SeriesDecomp(moving_avg_kernel_size)

        self._seasonal_layer = LinearBatchRelu(input_ch, 1024)
        self._trend_layer = LinearBatchRelu(input_ch, 1024)
        self._band_layer = nn.Linear(3 * lookback_days, 1024)

        self._dropout = nn.Dropout()
        self._relu_predictor = nn.LeakyReLU(0.1, inplace=True)
        self._instancenorm_predictor = nn.InstanceNorm1d(self._seq_len * 1024)
        self._linear_predictor = nn.Linear(self._seq_len * 1024, output_num)

    def _to_pred(self, output: torch.Tensor) -> torch.Tensor:
        output = output.view(output.shape[0], -1)
        output = self._relu_predictor(output)
        output = self._instancenorm_predictor(output)
        pred = self._linear_predictor(output).squeeze(-1)
        return pred

    def forward(self, past_inputs: PastInputs) -> torch.Tensor:
        candle_inputs = past_inputs.candle_inputs
        past_band_center = past_inputs.past_band_center
        past_band_upperbound = past_inputs.past_band_upperbound
        past_band_lowerbound = past_inputs.past_band_lowerbound

        seasonal_x, trend_x = self._decompose(candle_inputs)
        seasonal_x = self._seasonal_layer(seasonal_x)
        trend_x = self._trend_layer(trend_x)
        band_x = self._band_layer(torch.cat([past_band_center, past_band_lowerbound, past_band_upperbound], dim=-1))

        output = seasonal_x + trend_x + band_x.unsqueeze(-1)
        pred = self._to_pred(output)

        return pred
