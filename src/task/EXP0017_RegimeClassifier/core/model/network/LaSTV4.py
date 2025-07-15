import torch
from torch import nn

from ....util.registry import register_network


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
        self.TNet = t_func(
            in_dim, out_dim, seq_len, pred_len, inner_t, device=device, dropout=dropout
        )

    def forward(self, x_his, mean_inference):
        x_t, xt_rec, elbo_t, mlbo_t = self.TNet(x_his, mean_inference)

        rec_err = ((xt_rec - x_his) ** 2).mean()
        elbo = elbo_t - rec_err
        mlbo = mlbo_t

        return None, x_t, elbo, mlbo, None


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

        self.VarUnit_t = VarUnit(in_dim, inner_t)
        self.RecUnit_t = FeedNet(inner_t, in_dim, type="mlp", n_layers=1)

        self.t_pred_1 = FeedNet(self.seq_len, self.pred_len, type="mlp", n_layers=1)
        self.t_pred_2 = FeedNet(self.inner_t, self.out_dim, type="mlp", n_layers=1)

    def forward(self, x_his, mean_inference):
        # x_his = self.time_block(x_his)
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


@register_network("lastv4")
class LaSTV4(nn.Module):
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
        super(LaSTV4, self).__init__()
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

        _, x_t, elbo, mlbo, _ = self.LaSTLayer(x_his, mean_inference)
        x_pred = x_t
        x_pred = x_pred.squeeze(-1) if self.v_num > 1 else x_pred

        if self.norm_model:
            x_pred_denorm = self.norm_model(x_pred, "inverse")
        else:
            x_pred_denorm = x_pred

        if pass_norm:
            return x_pred, elbo, mlbo, None
        else:
            return x_pred_denorm, elbo, mlbo, None
