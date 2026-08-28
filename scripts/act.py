import torch
from torch import Tensor
from torch.nn import Module, Parameter, Transformer, TransformerEncoder, TransformerEncoderLayer, Linear
import numpy as np

class ACT(Module):
    def __init__(self, n_joints: int, seed: int | None):
        super().__init__()

        encoder_dim = 512
        decoder_dim = 512
        z_dim = 32

        self.rng = np.random.default_rng(seed=seed)
        self.image_encoder = TODO # resnet CNN
        self.chunk_encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model=encoder_dim, nhead=8, dim_feedforward=3200, dropout=0.1, batch_first=True),
            num_layers=4,
        ) # transformer encoder
        self.decoder = Transformer(
            d_model=decoder_dim, nhead=8, num_encoder_layers=4, num_decoder_layers=7, dim_feedforward=3200, dropout=0.1
        ) # transformer encoder-decoder
        self.latent_img_pos_embedding = TODO # sin-cos or learnable tensor
        self.latent_img_embedder = TODO # linear layer
        self.cls_embedding = Parameter(torch.randn(encoder_dim))  # TODO: Initialization magnitudes may be off
        self.angle_encoder_embedder = Linear(n_joints, encoder_dim)
        self.chunk_embedder = Linear(n_joints, encoder_dim)
        self.z_projector = Linear(encoder_dim, z_dim * 2)
        self.angle_decoder_embedder = Linear(n_joints, decoder_dim)
        self.z_embedder = Linear(z_dim, decoder_dim)

    def forward(
        self,
        img: Tensor,  # (batch_size, n_cameras, n_channels, height, width)
        angle: Tensor,  # (batch_size, n_joints)
        chunk: Tensor,  # (batch_size, chunk_len, n_joints)
    ):
        batch_size = img.size(0)

        angle_encoder_embedding: Tensor = self.angle_encoder_embedder(angle)  # (batch_size, encoder_dim)
        chunk_embedding: Tensor = self.chunk_embedder(chunk)  # (batch_size, chunk_length, encoder_dim)
        chunk_encoder_input = torch.cat(
            [
                self.cls_embedding.expand(batch_size, -1).unsqueeze(1),
                angle_encoder_embedding.unsqueeze(1),
                chunk_embedding
            ],
            dim=1,
        )
        latent_chunk: Tensor = self.chunk_encoder(chunk_encoder_input)
        z_mean, z_logvar = self.z_projector(latent_chunk[:, 0]).chunk(2, dim=-1)
        eps = self.rng.standard_normal(size=batch_size)
        z = z_mean + (z_logvar / 2).exp() * eps

        latent_img: Tensor = self.image_encoder(img)  # (batch_size, n_cameras, latent_depth, latent_height, latent_width)
        img_tokens = latent_img.permute(0, 1, 3, 4, 2)  # (batch_size, n_cameras, latent_height, latent_width, latent_depth)
        img_embeddings = (
            self.latent_img_embedder(img_tokens)  # (batch_size, n_cameras, latent_height, latent_width, decoder_dim)
            + self.latent_img_pos_embedding
        ).flatten(1, 3)  # (batch_size, n_cameras * latent_height * latent_width, decoder_dim)

        angle_decoder_embedding = self.angle_decoder_embedder(angle)  # (batch_size, decoder_dim)
        z_embedding = self.z_embedder(z)  # (batch_size, decoder_dim)
        decoder_input = torch.cat(
            [
                img_embeddings,
                angle_decoder_embedding.unsqueeze(1),
                z_embedding.unsqueeze(1)
            ],
            dim=1,
        )

        next_chunk = self.decoder(decoder_input, TODO)
        return next_chunk
