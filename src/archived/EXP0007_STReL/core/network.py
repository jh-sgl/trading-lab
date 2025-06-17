import numpy as np
import torch
import torch.nn as nn

from util.registry import register_network

from .const import Key


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
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self._in_ch = in_ch
        self._out_ch = out_ch

        self._linear = nn.Linear(in_ch, out_ch)
        self._norm = nn.InstanceNorm1d(out_ch)
        self._relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._linear(x)
        x = self._norm(x)
        x = self._relu(x)
        return x


@register_network("STReLUpstream")
class STReLUpstreamNetwork(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(
        self,
        input_ch: int,
        hidden_dim: int,
        lookback_num: int,
        lookahead_num: int,
        moving_avg_kernel_size: int,
    ) -> None:
        super().__init__()
        # TODO: revIN
        self._input_ch = input_ch
        self._hidden_dim = hidden_dim
        self._lookback_num = lookback_num
        self._lookahead_num = lookahead_num
        self._decompose = SeriesDecomp(moving_avg_kernel_size)

        self._input_norm = RevIN(input_ch)

        self._linear_seasonal_enc = nn.Linear(lookback_num * input_ch, hidden_dim)
        self._linear_trend_enc = nn.Linear(lookback_num * input_ch, hidden_dim)

        self._linear_seasonal_enc.weight = nn.Parameter(
            (1 / lookback_num) * torch.ones([hidden_dim, lookback_num * input_ch])
        )
        self._linear_trend_enc.weight = nn.Parameter(
            (1 / lookback_num) * torch.ones([hidden_dim, lookback_num * input_ch])
        )

        self._dropout = nn.Dropout()
        self._linear_predictor_past = nn.Linear(
            hidden_dim * input_ch, lookback_num * (5 + 1 + 2 + 4)
        )  # see dataset.__getitem__()
        self._linear_predictor_future = nn.Linear(
            hidden_dim * input_ch, lookahead_num * (5 + 1 + 2 + 4)
        )  # see dataset.__getitem__()

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def input_ch(self) -> int:
        return self._input_ch

    def _to_pred(self, output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = output.shape[0]
        output = output.view(batch_size, -1)
        output = self._dropout(output)
        pred_past = self._linear_predictor_past(output).squeeze(-1).view(batch_size, self._lookback_num, -1)
        pred_future = self._linear_predictor_future(output).squeeze(-1).view(batch_size, self._lookahead_num, -1)
        return pred_past, pred_future

    def _create_output_dict(self, pred: torch.Tensor) -> dict[Key, torch.Tensor]:
        outputs = {
            Key.REGRESSION: pred[..., :5],
            Key.CLASSIFICATION: pred[..., 5],
            Key.RECONSTRUCTION: pred[..., 6:8],
            Key.RANKING: pred[..., 8:],
        }
        return outputs

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self._input_norm(inputs, mode="norm")
        seasonal_x, trend_x = self._decompose(inputs)
        seasonal_output = self._linear_seasonal_enc(seasonal_x)
        trend_output = self._linear_trend_enc(trend_x)
        output = seasonal_output + trend_output
        output = self._input_norm(output, mode="rev")
        return output

    def forward(self, inputs: torch.Tensor) -> tuple[dict[Key, torch.Tensor], dict[Key, torch.Tensor]]:
        output = self.encode(inputs)
        pred_past, pred_future = self._to_pred(output)
        outputs_past = self._create_output_dict(pred_past)
        outputs_future = self._create_output_dict(pred_future)
        return outputs_past, outputs_future


@register_network("STReLUpstreamNonlinear")
class STReLUpstreamNonlinearNetwork(nn.Module):
    def __init__(
        self,
        input_ch: int,
        hidden_dim: int,
        lookback_num: int,
        lookahead_num: int,
        moving_avg_kernel_size: int,
    ) -> None:
        super().__init__()

        self._input_ch = input_ch
        self._hidden_dim = hidden_dim
        self._lookback_num = lookback_num
        self._lookahead_num = lookahead_num

        self._input_norm = RevIN(input_ch)
        self._decompose = SeriesDecomp(moving_avg_kernel_size)
        self._seasonal_layer = LinearBatchRelu(input_ch, hidden_dim)
        self._trend_layer = LinearBatchRelu(input_ch, hidden_dim)
        self._encode_layer_1 = LinearBatchRelu(hidden_dim, hidden_dim)
        self._encode_layer_2 = LinearBatchRelu(hidden_dim, hidden_dim)
        self._encode_layer_3 = LinearBatchRelu(hidden_dim, hidden_dim)

        self._dropout = nn.Dropout()
        self._linear_regr_past = nn.Sequential(
            LinearBatchRelu(hidden_dim, hidden_dim * input_ch), nn.Linear(hidden_dim * input_ch, 5)
        )  # see dataset.__getitem__()
        self._linear_clsf_past = nn.Sequential(
            LinearBatchRelu(hidden_dim, hidden_dim * input_ch), nn.Linear(hidden_dim * input_ch, 1)
        )  # see dataset.__getitem__()
        self._linear_reco_past = nn.Sequential(
            LinearBatchRelu(hidden_dim, hidden_dim * input_ch), nn.Linear(hidden_dim * input_ch, 2)
        )  # see dataset.__getitem__()
        self._linear_rank_past = nn.Sequential(
            LinearBatchRelu(hidden_dim, hidden_dim * input_ch), nn.Linear(hidden_dim * input_ch, 4)
        )  # see dataset.__getitem__()
        self._linear_regr_future = nn.Sequential(
            LinearBatchRelu(hidden_dim, hidden_dim * input_ch), nn.Linear(hidden_dim * input_ch, 5)
        )  # see dataset.__getitem__()
        self._linear_clsf_future = nn.Sequential(
            LinearBatchRelu(hidden_dim, hidden_dim * input_ch), nn.Linear(hidden_dim * input_ch, 1)
        )  # see dataset.__getitem__()
        self._linear_reco_future = nn.Sequential(
            LinearBatchRelu(hidden_dim, hidden_dim * input_ch), nn.Linear(hidden_dim * input_ch, 2)
        )  # see dataset.__getitem__()
        self._linear_rank_future = nn.Sequential(
            LinearBatchRelu(hidden_dim, hidden_dim * input_ch), nn.Linear(hidden_dim * input_ch, 4)
        )  # see dataset.__getitem__()

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def input_ch(self) -> int:
        return self._input_ch

    def _to_pred(self, output: torch.Tensor) -> tuple[dict[Key, torch.Tensor], dict[Key, torch.Tensor]]:
        output = self._dropout(output)
        pred_regr_past = self._linear_regr_past(output)
        pred_clsf_past = self._linear_clsf_past(output)
        pred_reco_past = self._linear_reco_past(output)
        pred_rank_past = self._linear_rank_past(output)
        pred_regr_future = self._linear_regr_future(output)
        pred_clsf_future = self._linear_clsf_future(output)
        pred_reco_future = self._linear_reco_future(output)
        pred_rank_future = self._linear_rank_future(output)

        return (
            {
                Key.REGRESSION: pred_regr_past,
                Key.CLASSIFICATION: pred_clsf_past.squeeze(-1),
                Key.RECONSTRUCTION: pred_reco_past,
                Key.RANKING: pred_rank_past,
            },
            {
                Key.REGRESSION: pred_regr_future,
                Key.CLASSIFICATION: pred_clsf_future.squeeze(-1),
                Key.RECONSTRUCTION: pred_reco_future,
                Key.RANKING: pred_rank_future,
            },
        )

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs = self._input_norm(inputs, mode="norm")
        inputs = inputs - inputs[:, 0:1]
        seasonal_x, trend_x = self._decompose(inputs)
        seasonal_x, trend_x = seasonal_x.permute(0, 2, 1), trend_x.permute(0, 2, 1)
        seasonal_output = self._seasonal_layer(seasonal_x)
        trend_output = self._trend_layer(trend_x)
        output = seasonal_output + trend_output
        output = self._encode_layer_1(output)
        output = self._encode_layer_2(output)
        output = self._encode_layer_3(output)
        # output = self._input_norm(output, mode="rev")
        return output

    def forward(self, inputs: torch.Tensor) -> tuple[dict[Key, torch.Tensor], dict[Key, torch.Tensor]]:
        output = self.encode(inputs)
        pred_past, pred_future = self._to_pred(output)
        return pred_past, pred_future


@register_network("STReLDownstream")
class STReLDownstreamNetwork(nn.Module):
    def __init__(self, input_ch, num_outputs, drop_prob):
        super().__init__()
        self._upstream_network: STReLUpstreamNetwork

        self._input_ch = input_ch
        self._num_outputs = num_outputs
        self._drop_prob = drop_prob

    def init_network(self, upstream_network: STReLUpstreamNetwork) -> None:
        self._upstream_network = upstream_network

        init_ = lambda m: self.init(
            m,
            torch.nn.init.orthogonal_,
            lambda x: torch.nn.init.constant_(x, 0),
            np.sqrt(2),
        )
        input_dim = self._upstream_network._lookahead_num * (5 + 1 + 2 + 4)
        self.linear = init_(torch.nn.Linear(input_dim, self._num_outputs))
        self.dropout = torch.nn.Dropout(p=self._drop_prob)
        self._rev_in = RevIN(self._input_ch)

    def init(self, module, weight_init, bias_init, gain=1):
        weight_init(module.weight.data, gain=gain)
        bias_init(module.bias.data)
        return module

    def forward(self, inputs):
        self._upstream_network.eval()
        encoded = self._upstream_network.encode(inputs)
        output_past, output_future = self._upstream_network._to_pred(encoded)
        output_future = torch.concat(
            [
                output_future[Key.REGRESSION],
                output_future[Key.CLASSIFICATION].unsqueeze(-1),
                output_future[Key.RECONSTRUCTION],
                output_future[Key.RANKING],
            ],
            dim=-1,
        )
        x = self.dropout(output_future.view(output_future.shape[0], -1))
        x = self.linear(x).squeeze()
        return x
