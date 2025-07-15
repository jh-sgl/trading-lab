import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ....util.registry import register_network


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, inputs):
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)

        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            - 2 * torch.matmul(flat_input, self.embedding.weight.t())
            + torch.sum(self.embedding.weight**2, dim=1)
        )

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self.num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)

        # Loss terms
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        return quantized, loss, encoding_indices.view(inputs.shape[0])


@register_network("signal_mlp_vq")
class SignalMLPVQ(nn.Module):
    def __init__(
        self,
        input_dim: int,
        repr_lookahead_num: int,
        latent_dim: int,
        num_codebook_vectors: int,
        output_dim: int,
        dropout_rate: float,
        commitment_cost: float = 0.25,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.repr_lookahead_num = repr_lookahead_num
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        # Encoder: maps flattened input to latent vector
        self.encoder = nn.Linear(input_dim * repr_lookahead_num, latent_dim)

        # VQ module
        self.vq = VectorQuantizer(
            num_embeddings=num_codebook_vectors,
            embedding_dim=latent_dim,
            commitment_cost=commitment_cost,
        )

        # Classifier MLP with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(latent_dim, output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """
        x: [batch_size, repr_lookahead_num, input_dim]
        """
        bs = x.shape[0]
        x_flat = x.view(bs, -1)
        z_e = self.encoder(x_flat)
        z_q, vq_loss, token_ids = self.vq(z_e)
        logits = self.classifier(z_q).softmax(dim=-1)
        return logits, vq_loss, token_ids
