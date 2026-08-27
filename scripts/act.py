import torch
from torch.nn import Transformer

class ACT:
    def __init__(self):
        self.image_encoder = TODO # resnet CNN
        self.chunk_encoder = TODO # transformer encoder
        self.decoder = TODO # transformer encoder-decoder
        self.latent_img_pos_embedding = TODO # sin-cos or learnable tensor
        self.latent_img_embedder = TODO # linear layer
        self.cls_embedding = TODO # learnable tensor
        self.angle_embedder = TODO # linear layer
        self.chunk_embedder = TODO # linear layer

    def train(
        self,
        img,  # (batch_size, n_cameras, n_channels, height, width)
        angle,  # (batch_size, n_joints)
        chunk,  # (batch_size, chunk_len, n_joints)
    ):
        latent_img: torch.Tensor = self.image_encoder(img)  # (batch_size, n_cameras, latent_depth, latent_height, latent_width)
        img_tokens = latent_img.permute(0, 1, 3, 4, 2)  # (batch_size, n_cameras, latent_height, latent_width, latent_depth)
        img_embeddings = (
            self.latent_img_embedder(img_tokens)  # (batch_size, n_cameras, latent_height, latent_width, decoder_dim)
            + self.latent_img_pos_embedding
        ).flatten(keep_dims=[0, 1, -1])  # (batch_size, n_cameras * latent_height * latent_width, decoder_dim)

        cls_embedding = self.cls_embedding  # (batch_size, encoder_dim)
        angle_embedding = self.angle_embedder(angle)  # (batch_size, encoder_dim)
        chunk_embedding = self.chunk_embedder(chunk)  # (batch_size, chunk_length, encoder_dim)
        encoder_input = torch.stack([cls_embedding.unsqueeze(1), angle_embedding.unsqueeze(1), chunk_embedding], dim=1)
