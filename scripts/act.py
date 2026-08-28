import torch
from torch import Tensor
from torch.nn import Module, Parameter, Transformer, TransformerEncoder, TransformerEncoderLayer, Linear
import numpy as np


class ACTEncoder(Module):
    def __init__(self, n_joints: int, d_model: int, z_dim: int):
        super().__init__()
        self.encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=3200, dropout=0.1, batch_first=True),
            num_layers=4,
        )
        self.cls_embedding = Parameter(torch.randn(d_model))  # TODO: Initialization magnitudes may be off
        self.angle_embedder = Linear(n_joints, d_model)
        self.chunk_embedder = Linear(n_joints, d_model)
        self.z_projector = Linear(d_model, z_dim * 2)

    def forward(
        self,
        angle: Tensor,  # (batch_size, n_joints)
        chunk: Tensor,  # (batch_size, chunk_len, n_joints)
    ) -> tuple[Tensor, Tensor]:
        batch_size = angle.size(0)
        angle_encoder_embedding: Tensor = self.angle_embedder(angle)  # (batch_size, encoder_dim)
        chunk_embedding: Tensor = self.chunk_embedder(chunk)  # (batch_size, chunk_length, encoder_dim)
        chunk_encoder_input = torch.cat(  # (batch_size, chunk_length, encoder_dim)
            [
                self.cls_embedding.expand(batch_size, -1).unsqueeze(1),
                angle_encoder_embedding.unsqueeze(1),
                chunk_embedding
            ],
            dim=1,
        )
        latent_chunk: Tensor = self.encoder(chunk_encoder_input)  # (batch_size, chunk_length, encoder_dim)
        z_mean, z_logvar = self.z_projector(latent_chunk[:, 0, :]).chunk(2, dim=-1)
        return z_mean, z_logvar


class ACTDecoder(Module):
    def __init__(self, n_joints: int, d_model: int, z_dim: int):
        self.decoder = Transformer(
            d_model=d_model, nhead=8, num_encoder_layers=4, num_decoder_layers=7, dim_feedforward=3200, dropout=0.1
        ) # transformer encoder-decoder
        self.latent_img_pos_embedding = TODO # sin-cos or learnable tensor
        self.latent_img_embedder = TODO # linear layer
        
        self.angle_decoder_embedder = Linear(n_joints, d_model)
        self.z_embedder = Linear(z_dim, d_model)

    def forward(
        self,
        latent_img: Tensor,  # (batch_size, n_cameras, latent_depth, latent_height, latent_width)
        angle: Tensor,  # (batch_size, n_joints)
        z: Tensor  # (batch_size, z_dim)
    ) -> Tensor:
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

        return self.decoder(decoder_input, TODO)


class ACT(Module):
    def __init__(self, n_joints: int, seed: int | None):
        super().__init__()
        self.z_dim = 32
        self.rng = np.random.default_rng(seed=seed)
        self.chunk_encoder = ACTEncoder(n_joints=n_joints, d_model=512, z_dim=self.z_dim)
        self.image_encoder = TODO # resnet CNN
        self.chunk_decoder = ACTDecoder(n_joints=n_joints, d_model=512, z_dim=self.z_dim)

    def forward(
        self,
        img: Tensor,  # (batch_size, n_cameras, n_channels, height, width)
        angle: Tensor,  # (batch_size, n_joints)
        chunk: Tensor | None,  # (batch_size, chunk_len, n_joints)
    ):
        batch_size = angle.size(0)
        if chunk is None:  # Inference
            z = torch.zeros(batch_size, self.z_dim)
        else:  # Training
            z_mean, z_logvar = self.chunk_encoder(angle, chunk)
            eps = self.rng.standard_normal(size=batch_size)
            z = z_mean + (z_logvar / 2).exp() * eps

        latent_img = self.image_encoder(img)
        next_chunk = self.chunk_decoder(latent_img, angle, z)
        return next_chunk, z_mean, z_logvar
