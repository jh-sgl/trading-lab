import torch
import torch.nn as nn
import torch.nn.functional as F
from alphasearch.learn.tslib.layers.Transformer_EncDec import Encoder, EncoderLayer
from alphasearch.learn.tslib.layers.SelfAttention_Family import (
    FullAttention,
    AttentionLayer,
)
from alphasearch.learn.tslib.layers.Embed import DataEmbedding_inverted
import numpy as np


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(
        self,
        tasks,
        enc_in,
        seq_len,
        pred_len,
        label_len,
        d_model,
        dropout,
        output_attention,
        n_heads,
        d_ff,
        activation,
        e_layers,
        factor,
    ):
        super(Model, self).__init__()
        self.tasks = tasks
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len
        self.output_attention = output_attention
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(
            c_in=seq_len,
            d_model=d_model,
            dropout=dropout,
        )
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=output_attention,
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
        )
        self.projection = nn.Linear(d_model, pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len :, :]  # [B, L, D]
