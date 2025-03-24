import torch
import torch.nn as nn


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
        moving_avg_kernel_size: int,
    ) -> None:
        super(DLinear, self).__init__()
        self._seq_len = seq_len
        self._pred_len = pred_len
        self._decompose = SeriesDecomp(moving_avg_kernel_size)

        self._batchnorm = nn.BatchNorm1d(self._seq_len)
        self._batchnorm_seasonal = nn.BatchNorm1d(input_ch)
        self._batchnorm_trend = nn.BatchNorm1d(input_ch)

        self._linear_seosonal_enc = nn.Linear(self._seq_len, self._pred_len)
        self._linear_trend_enc = nn.Linear(self._seq_len, self._pred_len)

        self._linear_seosonal_enc.weight = nn.Parameter(
            (1 / self._seq_len) * torch.ones([self._pred_len, self._seq_len])
        )
        self._linear_trend_enc.weight = nn.Parameter((1 / self._seq_len) * torch.ones([self._pred_len, self._seq_len]))
        self._dropout = nn.Dropout()
        self._linear_predictor = nn.Linear(self._pred_len * input_ch, 3)

    def _to_pred(self, output: torch.Tensor) -> torch.Tensor:
        output = output.view(output.shape[0], -1)
        output = self._dropout(output)
        pred = self._linear_predictor(output).squeeze()
        return pred

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seasonal_x, trend_x = self._decompose(x)

        seasonal_output = self._linear_seosonal_enc(seasonal_x)
        trend_output = self._linear_trend_enc(trend_x)
        output = seasonal_output + trend_output

        pred = self._to_pred(output)
        return pred
