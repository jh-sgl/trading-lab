import torch
import torch.nn as nn
import torch.nn.functional as F

from ....util.registry import register_network


@register_network("signal_causal_aware_classifier")
class SignalCausalAwareClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        time_len: int,
        output_dim: int,
        dropout_rate: float,
        regime_token: bool = True,
        proj_dim: int = 64,
        heads: int = 4,
        use_projection: bool = False,
        mask_prob: float = 0.2,
        mask_features: bool = True,
        mask_noise_std: float = 0.01,
        pos_embed: bool = True,
        num_envs: int = 4,
        disentangle_dim: int = 32,
        learn_feature_graph: bool = True,
    ):
        super().__init__()
        self.time_len = time_len
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mask_prob = mask_prob
        self.use_projection = use_projection
        self.regime_token = regime_token
        self.mask_features = mask_features
        self.mask_noise_std = mask_noise_std
        self.pos_embed_enabled = pos_embed
        self.num_envs = num_envs
        self.learn_feature_graph = learn_feature_graph

        self.seq_len = time_len + int(regime_token)

        # Main attention backbone
        self.attn = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=heads, batch_first=True
        )
        self.dropout = nn.Dropout(p=dropout_rate)

        if regime_token:
            self.regime_token_embed = nn.Parameter(torch.randn(1, 1, input_dim))

        if self.pos_embed_enabled:
            self.pos_embed = nn.Parameter(torch.randn(1, self.seq_len, input_dim))

        self.pool_token = nn.Parameter(torch.randn(1, 1, input_dim))
        self.pool_proj = nn.Linear(input_dim, output_dim)

        if use_projection:
            self.projector = nn.Sequential(
                nn.Linear(input_dim, proj_dim),
                nn.ReLU(),
                nn.Linear(proj_dim, proj_dim),
            )

        # Environment embedding
        self.env_embed = nn.Linear(input_dim, num_envs)

        # Disentangle head for causal path probing
        self.disentangle_head = nn.Sequential(
            nn.Linear(input_dim, disentangle_dim),
            nn.ReLU(),
            nn.Linear(disentangle_dim, input_dim),
        )

        # Optional learnable adjacency over features
        if learn_feature_graph:
            self.feature_adj = nn.Parameter(torch.randn(input_dim, input_dim))  # (F, F)

    def _causal_mask(self, size):
        return (
            torch.tril(torch.ones(size, size)).bool().to(next(self.parameters()).device)
        )

    def _intervention_mask(self, x):
        """
        Randomly mask features or time steps with given probability.
        If `mask_features` is True, randomly mask features independently.
        Otherwise, mask entire time steps.
        """
        B, T, Feat = x.size()
        x_masked = x.clone()

        if self.mask_features:
            mask = torch.rand_like(x_masked) < self.mask_prob
        else:
            time_mask = (torch.rand(B, T, 1, device=x.device) < self.mask_prob).expand(
                -1, -1, Feat
            )
            mask = time_mask

        noise = torch.randn_like(x_masked) * self.mask_noise_std
        x_masked[mask] = noise[mask]

        return x_masked

    def _add_positional_embed(self, x):
        if self.pos_embed_enabled:
            x = x + self.pos_embed[:, : x.size(1), :]
        return x

    def _apply_feature_graph(self, x):
        if not self.learn_feature_graph:
            return x
        # Apply linear feature graph propagation
        B, T, Feat = x.shape
        adj = torch.sigmoid(self.feature_adj)
        x_graph = torch.matmul(x, adj)  # (B, T, Feat) x (Feat, Feat) -> (B, T, Feat)
        return x + x_graph  # residual

    def forward(self, x, compute_consistency=False):
        """
        x: (B, T, Feat)
        """
        B, T, Feat = x.size()
        assert T == self.time_len and Feat == self.input_dim

        x = self._apply_feature_graph(x)

        if self.regime_token:
            token = self.regime_token_embed.expand(B, -1, -1)
            x_input = torch.cat([token, x], dim=1)
        else:
            x_input = x

        x_input = self._add_positional_embed(x_input)

        causal_mask = self._causal_mask(x_input.shape[1])
        attn_out, attn_weights = self.attn(
            x_input, x_input, x_input, attn_mask=~causal_mask
        )

        # Attention pooling
        pool_query = self.pool_token.expand(B, -1, -1)
        pooled, _ = self.attn(pool_query, attn_out, attn_out)
        pooled = self.dropout(pooled.squeeze(1))

        logits = self.pool_proj(pooled)
        preds = F.softmax(logits, dim=-1)

        # Environment prediction (unsupervised domain ID)
        # env_logits = self.env_embed(pooled)

        # Causal disentangling head (representation stability)
        # disentangled = self.disentangle_head(pooled)

        if not compute_consistency:
            return preds

        # Intervened forward pass
        x_masked = self._intervention_mask(x)
        x_masked = self._apply_feature_graph(x_masked)

        if self.regime_token:
            token = self.regime_token_embed.expand(B, 1, Feat)
            x_masked_input = torch.cat([token, x_masked], dim=1)
        else:
            x_masked_input = x_masked

        x_masked_input = self._add_positional_embed(x_masked_input)

        attn_out_masked, _ = self.attn(
            x_masked_input, x_masked_input, x_masked_input, attn_mask=~causal_mask
        )
        pooled_masked, _ = self.attn(pool_query, attn_out_masked, attn_out_masked)
        pooled_masked = self.dropout(pooled_masked.squeeze(1))

        logits_masked = self.pool_proj(pooled_masked)
        preds_masked = F.softmax(logits_masked, dim=-1)

        # KL divergence loss
        kl_div = F.kl_div(preds_masked.log(), preds, reduction="batchmean")

        return preds, kl_div

    def get_representation(self, x):
        """Returns pooled representation for contrastive or causal probing."""
        B, T, Feat = x.size()
        x = self._apply_feature_graph(x)

        if self.regime_token:
            token = self.regime_token_embed.expand(B, -1, -1)
            x_input = torch.cat([token, x], dim=1)
        else:
            x_input = x

        x_input = self._add_positional_embed(x_input)
        causal_mask = self._causal_mask(x_input.shape[1])
        attn_out, _ = self.attn(x_input, x_input, x_input, attn_mask=~causal_mask)

        pool_query = self.pool_token.expand(B, -1, -1)
        pooled, _ = self.attn(pool_query, attn_out, attn_out)
        pooled = pooled.squeeze(1)

        return self.projector(pooled) if self.use_projection else pooled
