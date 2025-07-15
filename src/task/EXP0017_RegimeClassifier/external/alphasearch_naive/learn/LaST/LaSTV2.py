import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.fft import rfft, irfft

_NEXT_FAST_LEN = {}


def next_fast_len(size):
    """
    Returns the next largest number ``n >= size`` whose prime factors are all
    2, 3, or 5. These sizes are efficient for fast fourier transforms.
    Equivalent to :func:`scipy.fftpack.next_fast_len`.
    :param int size: A positive number.
    :returns: A possibly larger number.
    :rtype int:
    """
    try:
        return _NEXT_FAST_LEN[size]
    except KeyError:
        pass

    assert isinstance(size, int) and size > 0
    next_size = size
    while True:
        remaining = next_size
        for n in (2, 3, 5):
            while remaining % n == 0:
                remaining //= n
        if remaining == 1:
            _NEXT_FAST_LEN[size] = next_size
            return next_size
        next_size += 1


def topk_mask(input, k, dim, device, return_mask=False):
    topk, indices = torch.topk(input, k, dim=dim)
    fill = 1 if return_mask else topk
    masked = torch.zeros_like(input, device=device).scatter_(dim, indices, fill)
    # vals, idx = input.topk(k, dim=dim)
    # topk = torch.zeros_like(input)
    # topk[idx] = 1 if return_mask else vals
    return masked


def autocorrelation(input, dim=0):
    """
    Computes the autocorrelation of samples at dimension ``dim``.
    Reference: https://en.wikipedia.org/wiki/Autocorrelation#Efficient_computation
    """
    if (not input.is_cuda) and (not torch.backends.mkl.is_available()):
        raise NotImplementedError(
            "For CPU tensor, this method is only supported " "with MKL installed."
        )

    # https://github.com/stan-dev/math/blob/develop/stan/math/prim/mat/fun/autocorrelation.hpp
    N = input.size(dim)
    M = next_fast_len(N)
    M2 = 2 * M

    # transpose dim with -1 for Fourier transform
    input = input.transpose(dim, -1)

    centered_signal = input - input.mean(dim=-1, keepdim=True)

    # Fourier transform
    freqvec = torch.view_as_real(rfft(centered_signal, n=M2))
    # take square of magnitude of freqvec (or freqvec x freqvec*)
    freqvec_gram = freqvec.pow(2).sum(-1)
    # inverse Fourier transform
    autocorr = irfft(freqvec_gram, n=M2)

    # truncate and normalize the result, then transpose back to original shape
    autocorr = autocorr[..., :N]
    autocorr = autocorr / torch.tensor(
        range(N, 0, -1), dtype=input.dtype, device=input.device
    )
    autocorr = autocorr / autocorr[..., :1]
    return autocorr.transpose(dim, -1)


def log_Normal_diag(x, mean, var, average=True, dim=None):
    log_normal = -0.5 * (torch.log(var) + torch.pow(x - mean, 2) / var**2)
    if average:
        return torch.mean(log_normal, dim)
    else:
        return torch.sum(log_normal, dim)


def log_Normal_standard(x, average=True, dim=None):
    log_normal = -0.5 * torch.pow(x, 2)
    if average:
        return torch.mean(log_normal, dim)
    else:
        return torch.sum(log_normal, dim)


def shape_extract(x, device, mode="value"):
    x_diff = x[:, 1:] - x[:, :-1]
    if mode == "binary":
        x_diff[x_diff > 0] = 1.0
        x_diff[x_diff <= 0] = 0.0
        x_diff = x_diff.type(torch.LongTensor).to(device)
    return x_diff


def period_sim(x, y):
    x = x.reshape(-1, x.shape[1])
    y = y.reshape(-1, y.shape[1])
    # input size: batch x length
    """ Autocorrelation """
    x_ac = autocorrelation(x, dim=1)[:, 1:]
    y_ac = autocorrelation(y, dim=1)[:, 1:]

    distance = ((x_ac - y_ac) ** 2).mean(dim=1).mean()

    return -distance


def trend_sim(x, y, device):
    # input size: batch x length
    x = x.reshape(-1, x.shape[1])
    y = y.reshape(-1, y.shape[1])
    x_t = shape_extract(x, device)
    y_t = shape_extract(y, device)

    """ The First Order Temporal Correlation Coefficient (CORT) """
    denominator = torch.sqrt(torch.pow(x_t, 2).sum(dim=1)) * torch.sqrt(
        torch.pow(y_t, 2).sum(dim=1)
    )
    numerator = (x_t * y_t).sum(dim=1)
    cort = numerator / denominator

    return cort.mean()


class NonLinear(nn.Module):
    def __init__(self, input_size, output_size, bias=True, activation=None):
        super(NonLinear, self).__init__()

        self.activation = activation
        self.linear = nn.Linear(input_size, output_size, bias=bias)

    def forward(self, x):
        h = self.linear(x)
        if self.activation is not None:
            h = self.activation(h)

        return h


class CriticFunc(nn.Module):
    def __init__(self, x_dim, y_dim, dropout=0.1):
        super(CriticFunc, self).__init__()
        cat_dim = x_dim + y_dim
        self.critic = nn.Sequential(
            nn.Linear(cat_dim, cat_dim // 4),
            nn.ReLU(),
            nn.Linear(cat_dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, y):
        cat = torch.cat((x, y), dim=-1)
        return self.critic(cat)


class NeuralFourierLayer(nn.Module):
    def __init__(self, in_dim, out_dim, seq_len=168, pred_len=24, device=None):
        super().__init__()

        self.out_len = seq_len + pred_len
        self.freq_num = (seq_len // 2) + 1

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device

        self.weight = nn.Parameter(
            torch.empty((self.freq_num, in_dim, out_dim), dtype=torch.cfloat)
        )
        self.bias = nn.Parameter(
            torch.empty((self.freq_num, out_dim), dtype=torch.cfloat)
        )
        self.init_parameters()

    def forward(self, x_emb, mask=False):
        # input - b t d
        x_fft = rfft(x_emb, dim=1)[:, : self.freq_num]
        # output_fft = torch.einsum('bti,tio->bto', x_fft.type_as(self.weight), self.weight) + self.bias
        output_fft = x_fft
        if mask:
            amp = output_fft.abs().permute(0, 2, 1).reshape((-1, self.freq_num))
            output_fft_mask = topk_mask(
                amp, k=8, dim=1, device=self.device, return_mask=True
            )
            output_fft_mask = output_fft_mask.reshape(
                x_emb.shape[0], self.out_dim, self.freq_num
            ).permute(0, 2, 1)
            output_fft = output_fft * output_fft_mask
        return irfft(output_fft, n=self.out_len, dim=1)

    def init_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)


class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i)
            )
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(
        self, seq_len, pred_len, device, d_model=32, d_ff=32, top_k=5, num_kernels=6
    ):
        super(TimesBlock, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = top_k
        self.device = device
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model, num_kernels=num_kernels),
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros(
                    [x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]
                ).to(self.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len + self.pred_len
                out = x
            # reshape
            out = (
                out.reshape(B, length // period, period, N)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, : (self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class FeedNet(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        type="mlp",
        n_layers=1,
        inner_dim=None,
        activaion=None,
        dropout=0.1,
    ):
        super(FeedNet, self).__init__()
        self.n_layers = n_layers
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.type = type
        self.dropout = dropout

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            layer_in = in_dim if i == 0 else inner_dim[i - 1]
            layer_out = out_dim if i == n_layers - 1 else inner_dim[i]
            if type == "mlp":
                self.layers.append(nn.Linear(layer_in, layer_out))
            else:
                raise Exception(
                    "KeyError: Feedward Net keyword error. Please use word in ['mlp']"
                )
            if i != n_layers - 1 and activaion is not None:
                self.layers.append(activaion)

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x


class VarUnit(nn.Module):
    def __init__(self, in_dim, z_dim):
        super(VarUnit, self).__init__()

        self.in_dim = in_dim
        self.z_dim = z_dim

        self.loc_net = FeedNet(in_dim, z_dim, type="mlp", n_layers=1)
        self.var_net = nn.Sequential(
            FeedNet(in_dim, z_dim, type="mlp", n_layers=1),
            nn.Softplus(),
        )

        self.critic_xz = CriticFunc(z_dim, in_dim)

    def normal_init(self, m, mean=0.0, std=0.01):
        m.weight.data.normal_(mean, std)

    def log_p_z(self, z):
        log_prior = log_Normal_standard(z, dim=1)
        return log_prior

    def compute_KL(self, z_q, z_q_mean, z_q_var):
        log_p_z = self.log_p_z(z_q)
        log_q_z = log_Normal_diag(z_q, z_q_mean, z_q_var, dim=1)
        KL = -(log_p_z - log_q_z)

        return KL.mean()

    def compute_MLBO(self, x, z_q, method="our"):
        idx = torch.randperm(z_q.shape[0])
        z_q_shuffle = z_q[idx].view(z_q.size())
        if method == "MINE":
            mlbo = (
                self.critic_xz(x, z_q).mean()
                - torch.log(
                    torch.exp(self.critic_xz(x, z_q_shuffle))
                    .squeeze(dim=-1)
                    .mean(dim=-1)
                ).mean()
            )
        else:
            point = 1 / torch.exp(self.critic_xz(x, z_q_shuffle)).squeeze(dim=-1).mean()
            point = point.detach()

            if len(x.shape) == 3:
                mlbo = self.critic_xz(x, z_q) - point * torch.exp(
                    self.critic_xz(x, z_q_shuffle)
                )  # + 1 + torch.log(point)
            else:
                mlbo = self.critic_xz(x, z_q) - point * torch.exp(
                    self.critic_xz(x, z_q_shuffle)
                )

        return mlbo.mean()

    def forward(self, x, return_para=True):
        mean, var = self.loc_net(x), self.var_net(x)
        # var = torch.abs(var)
        qz_gaussian = torch.distributions.Normal(loc=mean, scale=var)
        qz = qz_gaussian.rsample()  # mu+sigma*epsilon
        return (qz, mean, var) if return_para else qz


class CalculateMubo(nn.Module):
    def __init__(self, x_dim, y_dim, dropout=0.1):
        super().__init__()
        self.critic_st = CriticFunc(x_dim, y_dim, dropout)

    def forward(self, x_his, var_net_t, var_net_s):
        zs, zt = var_net_s(x_his, return_para=False), var_net_t(
            x_his, return_para=False
        )
        idx = torch.randperm(zt.shape[0])
        zt_shuffle = zt[idx].view(zt.size())
        f_st = self.critic_st(zs, zt)
        f_s_t = self.critic_st(zs, zt_shuffle)

        mubo = f_st - f_s_t
        pos_mask = torch.zeros_like(f_st)
        pos_mask[mubo < 0] = 1
        mubo_musk = mubo * pos_mask
        reg = (mubo_musk**2).mean()

        return mubo.mean() + reg


class LaSTBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        seq_len,
        pred_len,
        s_func,
        inner_s,
        t_func,
        inner_t,
        device,
        dropout=0.1,
    ):
        super().__init__()
        self.input_dim = in_dim
        self.out_dim = out_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.SNet = s_func(
            in_dim, out_dim, seq_len, pred_len, inner_s, device=device, dropout=dropout
        )
        self.TNet = t_func(
            in_dim, out_dim, seq_len, pred_len, inner_t, device=device, dropout=dropout
        )
        self.MuboNet = CalculateMubo(inner_s, inner_t, dropout=dropout)

    def forward(self, x_his, mean_inference):
        x_s, xs_rec, elbo_s, mlbo_s = self.SNet(x_his, mean_inference)
        x_t, xt_rec, elbo_t, mlbo_t = self.TNet(x_his, mean_inference)

        rec_err = ((xs_rec + xt_rec - x_his) ** 2).mean()
        elbo = elbo_t + elbo_s - rec_err
        mlbo = mlbo_t + mlbo_s
        mubo = self.MuboNet(x_his, self.SNet.VarUnit_s, self.TNet.VarUnit_t)

        return x_s, x_t, elbo, mlbo, mubo


class SNet(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        seq_len,
        pred_len,
        inner_s,
        device,
        dropout=0.1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.inner_s = inner_s
        self.device = device

        """ Time Block"""
        self.time_block = TimesBlock(
            self.seq_len, 0, device=device, d_model=in_dim, d_ff=in_dim
        )

        """ VAE Net """
        self.VarUnit_s = VarUnit(in_dim, inner_s)
        self.rec_s = nn.Linear(inner_s, in_dim)
        self.RecUnit_s = FeedNet(
            inner_s, in_dim, type="mlp", n_layers=1, dropout=dropout
        )
        """ Fourier """
        self.FourierNet = NeuralFourierLayer(
            inner_s, out_dim, seq_len, pred_len, device=self.device
        )
        self.pred = FeedNet(self.inner_s, self.out_dim, type="mlp", n_layers=1)

    def forward(self, x_his, mean_inference):
        x_his = self.time_block(x_his)
        qz_s, mean_qz_s, var_qz_s = self.VarUnit_s(x_his)
        xs_rec = self.RecUnit_s(qz_s)
        elbo_s = period_sim(xs_rec, x_his) - self.VarUnit_s.compute_KL(
            qz_s, mean_qz_s, var_qz_s
        )
        mlbo_s = self.VarUnit_s.compute_MLBO(x_his, qz_s)

        """ Fourier """
        if mean_inference:
            xs_pred = self.pred(self.FourierNet(mean_qz_s)[:, -self.pred_len :])
        else:
            xs_pred = self.pred(self.FourierNet(qz_s)[:, -self.pred_len :])

        return xs_pred, xs_rec, elbo_s, mlbo_s


class TNet(nn.Module):
    def __init__(
        self, in_dim, out_dim, seq_len, pred_len, inner_t, device, dropout=0.1
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.inner_t = inner_t
        self.device = device

        """ Time Block"""
        self.time_block = TimesBlock(
            self.seq_len, 0, device=device, d_model=in_dim, d_ff=in_dim
        )

        self.VarUnit_t = VarUnit(in_dim, inner_t)
        self.RecUnit_t = FeedNet(inner_t, in_dim, type="mlp", n_layers=1)

        self.t_pred_1 = FeedNet(self.seq_len, self.pred_len, type="mlp", n_layers=1)
        self.t_pred_2 = FeedNet(self.inner_t, self.out_dim, type="mlp", n_layers=1)

    def forward(self, x_his, mean_inference):
        x_his = self.time_block(x_his)
        qz_t, mean_qz_t, var_qz_t = self.VarUnit_t(x_his)
        xt_rec = self.RecUnit_t(qz_t)
        elbo_t = trend_sim(xt_rec, x_his, self.device) - self.VarUnit_t.compute_KL(
            qz_t, mean_qz_t, var_qz_t
        )
        mlbo_t = self.VarUnit_t.compute_MLBO(x_his, qz_t)

        # mlp
        if len(x_his.shape) == 3:
            if mean_inference:
                xt_pred = self.t_pred_2(
                    self.t_pred_1(mean_qz_t.permute(0, 2, 1)).permute(0, 2, 1)
                )
            else:
                xt_pred = self.t_pred_2(
                    self.t_pred_1(qz_t.permute(0, 2, 1)).permute(0, 2, 1)
                )
        else:
            if mean_inference:
                xt_pred = self.t_pred_2(
                    self.t_pred_1(mean_qz_t.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
                )
            else:
                xt_pred = self.t_pred_2(
                    self.t_pred_1(qz_t.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
                )

        return xt_pred, xt_rec, elbo_t, mlbo_t


class LaSTV2(nn.Module):
    def __init__(
        self,
        input_len,
        output_len,
        input_dim,
        out_dim,
        device,
        var_num=1,
        latent_dim=64,
        dropout=0.1,
        norm_model=None,
        inner_norm=None,
    ):
        super(LaSTV2, self).__init__()
        self.in_dim = input_dim
        self.out_dim = out_dim
        self.seq_len = input_len
        self.pred_len = output_len

        self.v_num = var_num
        self.inner_s = latent_dim
        self.inner_t = latent_dim
        self.dropout = dropout
        self.device = device
        self.norm_model = norm_model
        self.inner_norm = inner_norm

        self.LaSTLayer = LaSTBlock(
            self.in_dim,
            self.out_dim,
            input_len,
            output_len,
            SNet,
            self.inner_s,
            TNet,
            self.inner_t,
            self.device,
            dropout=dropout,
        )

    def forward(self, x, x_mark=None, mean_inference=False, pass_norm=False, **kwargs):
        if self.norm_model:
            x, _ = self.norm_model(x, "forward")

        b, t, _ = x.shape
        x_his = x
        if self.v_num > 1:
            x_his = x_his.reshape(b, t, self.v_num, -1)
            if x_mark is not None:
                x_his = torch.cat(
                    [x_his, x_mark.unsqueeze(dim=2).repeat(1, 1, self.v_num, 1)], dim=-1
                )
        else:
            if x_mark is not None:
                x_his = torch.cat([x_his, x_mark], dim=-1)

        x_s, x_t, elbo, mlbo, mubo = self.LaSTLayer(x_his, mean_inference)
        x_pred = x_s + x_t
        x_pred = x_pred.squeeze(-1) if self.v_num > 1 else x_pred

        if self.norm_model:
            x_pred_denorm = self.norm_model(x_pred, "inverse")
        else:
            x_pred_denorm = x_pred

        if pass_norm:
            return x_pred, elbo, mlbo, mubo
        else:
            return x_pred_denorm, elbo, mlbo, mubo
