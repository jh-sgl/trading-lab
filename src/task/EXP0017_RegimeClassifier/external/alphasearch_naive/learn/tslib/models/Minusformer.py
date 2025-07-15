import torch
import torch.nn as nn
import torch.nn.functional as F
from alphasearch.learn.tslib.layers.SelfAttention_Family import (
    FullAttention,
    AttentionLayer,
    FlashAttention,
    ProbAttention,
)
from alphasearch.learn.tslib.layers.Embed import DataEmbedding_inverted
import numpy as np


class standard_scaler:
    def __init__(self, ts, sub_last=False, cat_std=False):
        self.sub_last = sub_last
        self.cat_std = cat_std
        self.mean = ts.mean(-1, keepdim=True)
        self.std = torch.sqrt(
            torch.var(ts - self.mean, dim=-1, keepdim=True, unbiased=False) + 1e-5
        )

    def transform(self, data):
        if self.sub_last:
            self.last_value = data[..., -1:].detach()
            data = data - self.last_value
        data = (data - self.mean) / self.std
        if self.cat_std:
            data = torch.cat((data, self.mean, self.std), -1)
        return data

    def inverted(self, data):
        if self.cat_std:
            data = data[..., :-2] * data[..., -1:] + data[..., -2:-1]
        else:
            data = (data * self.std) + self.mean
        data = data + self.last_value if self.sub_last else data
        return data


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_block, d_ff=None, dropout=0.1, gate=1):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.conv3 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.conv4 = nn.Conv1d(d_model, d_model, kernel_size=1)
        d_ff = d_model * 2 if attention else d_model
        self.conv5 = nn.Conv1d(d_ff, d_block, kernel_size=1)
        self.conv6 = nn.Conv1d(d_ff, d_block, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = F.gelu if attention else F.relu
        self.gate = gate

        print("Layer --- ")

    def forward(self, x, attn_mask=None):

        if self.attention:
            x_att, _ = self.attention(x, x, x, attn_mask=attn_mask)
            x = x - self.dropout(x_att)

        x_ln = x = self.norm1(x)
        x_ln = self.dropout(self.act(self.conv1(x_ln.transpose(-1, 1))))
        x_ln = self.dropout(self.conv2(x_ln).transpose(-1, 1))

        x = (x - x_ln).transpose(-1, 1)
        h_gate = F.sigmoid(self.conv3(x)) if self.gate else 1
        h = h_gate * self.conv4(x)

        out = torch.cat((x_att, x_ln), -1) if self.attention else x_ln
        out = out.transpose(-1, 1)
        gate = F.sigmoid(self.conv5(out)) if self.gate else 1
        out = gate * self.conv6(out)

        return self.norm2(h.transpose(-1, 1)), out.transpose(-1, 1)


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = (
            nn.ModuleList(conv_layers) if conv_layers is not None else None
        )

    def forward(self, x, attn_mask=None):

        output = 0
        for attn_layer in self.attn_layers:
            x, out = attn_layer(x, attn_mask=attn_mask)
            output = out - output

        return output


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2402.02332
    """

    def __init__(
        self,
        seq_len,
        pred_len,
        label_len,
        d_block,
        d_model,
        d_ff,
        n_heads,
        factor,
        dropout,
        e_layers,
        gate,
        attn,
        output_attention,
    ):
        super(Model, self).__init__()
        self.pred_len = pred_len
        self.label_len = label_len
        self.embed = nn.Linear(seq_len, d_model)
        self.backbone = Encoder(
            [
                EncoderLayer(
                    (
                        AttentionLayer(
                            FullAttention(
                                False,
                                factor,
                                attention_dropout=dropout,
                                output_attention=output_attention,
                            ),
                            d_model,
                            n_heads,
                        )
                        if attn
                        else None
                    ),
                    d_model,
                    d_block,
                    d_ff,
                    dropout=dropout,
                    gate=gate,
                )
                for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
        )
        if d_block != pred_len:
            self.align = nn.Linear(d_block, pred_len)
        else:
            self.align = nn.Identity()

        print("Minusformer ...")

    def forward(self, x, x_mark, x_dec, x_mark_dec, mask=None):
        # x : (Batch, Seq_len, Dim)
        x = x.permute(0, 2, 1)
        scaler = standard_scaler(x)
        x = scaler.transform(x)
        if x_mark is not None:
            x_emb = self.embed(torch.cat((x, x_mark.permute(0, 2, 1)), 1))
        else:
            x_emb = self.embed(x)
        output = self.backbone(x_emb)
        output = self.align(output)
        output = scaler.inverted(output[:, : x.size(1), :])
        return output.permute(0, 2, 1)
