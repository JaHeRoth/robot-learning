import torch
from torch import Tensor
from torch.nn import Module, Transformer, TransformerEncoder, TransformerEncoderLayer, Linear, Embedding, Sequential
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.transforms import Normalize


def _make_sequence_pos_embedding(length: int, dim: int):
    assert dim % 2 == 0
    T = 10000
    w = 1 / T ** (2 * torch.arange(dim // 2) / dim)
    embeddings = [
        torch.stack([(i * w).sin(), (i * w).cos()]).permute(1, 0).flatten()
        for i in range(length)
    ]
    return torch.stack(embeddings)


class ACTEncoder(Module):
    def __init__(self, action_dim: int, d_model: int, z_dim: int, chunk_len: int):
        super().__init__()
        self.cls_embedding = Embedding(num_embeddings=1, embedding_dim=d_model)
        self.proprio_embedder = Linear(action_dim, d_model)
        self.chunk_embedder = Linear(action_dim, d_model)
        self.register_buffer(
            name="pos_embedding",
            tensor=_make_sequence_pos_embedding(
                length=chunk_len + 2, dim=d_model
            )
        )
        self.encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=3200, dropout=0.1, batch_first=True),
            num_layers=4,
        )
        self.z_projector = Linear(d_model, z_dim * 2)

    def forward(
        self,
        proprio: Tensor,  # (batch_size, action_dim)
        chunk: Tensor,  # (batch_size, chunk_len, action_dim)
    ) -> tuple[Tensor, Tensor]:
        batch_size = proprio.size(0)
        proprio_encoder_embedding: Tensor = self.proprio_embedder(proprio)  # (batch_size, encoder_dim)
        chunk_embedding: Tensor = self.chunk_embedder(chunk)  # (batch_size, chunk_length, encoder_dim)
        chunk_encoder_input = torch.cat(  # (batch_size, chunk_length + 2, encoder_dim)
            [
                self.cls_embedding.weight.expand(batch_size, -1).unsqueeze(1),
                proprio_encoder_embedding.unsqueeze(1),
                chunk_embedding
            ],
            dim=1,
        ) + self.pos_embedding
        latent_chunk: Tensor = self.encoder(chunk_encoder_input)  # (batch_size, chunk_length + 2, encoder_dim)
        z_mean, z_logvar = self.z_projector(latent_chunk[:, 0, :]).chunk(2, dim=-1)
        return z_mean, z_logvar


def _make_img_tokens_pos_embedding(rows: int, cols: int, dim: int):
    assert dim % 4 == 0
    D = dim // 2
    T = 10000
    w = 1 / T ** (2 * torch.arange(D // 2) / D)
    embeddings = []
    for row in range(rows):
        for col in range(cols):
            row_embedding = torch.stack([(row * w).sin(), (row * w).cos()]).permute(1, 0).flatten()
            col_embedding = torch.stack([(col * w).sin(), (col * w).cos()]).permute(1, 0).flatten()
            embeddings.append(torch.cat([row_embedding, col_embedding]))
    return torch.stack(embeddings).reshape(rows, cols, dim)


class ACTDecoder(Module):
    def __init__(self, action_dim: int, d_model: int, z_dim: int, chunk_len: int, img_tokens_shape: tuple):
        super().__init__()
        self.register_buffer(
            name="img_tokens_pos_embedding",
            tensor=_make_img_tokens_pos_embedding(
                rows=img_tokens_shape[0], cols=img_tokens_shape[1], dim=d_model
            )
        )
        self.img_tokens_embedder = Linear(img_tokens_shape[2], d_model)
        self.proprio_embedder = Linear(action_dim, d_model)
        self.z_embedder = Linear(z_dim, d_model)
        self.decoder_decoder_pos_embedding = Embedding(num_embeddings=chunk_len, embedding_dim=d_model)
        self.decoder = Transformer(
            d_model=d_model, nhead=8, num_encoder_layers=4, num_decoder_layers=7, dim_feedforward=3200, dropout=0.1, batch_first=True
        )
        self.action_projector = Linear(d_model, action_dim)

    def forward(
        self,
        latent_imgs: Tensor,  # (batch_size, n_cameras, latent_depth, latent_height, latent_width)
        proprio: Tensor,  # (batch_size, action_dim)
        z: Tensor,  # (batch_size, z_dim)
    ) -> Tensor:
        img_tokens = latent_imgs.permute(0, 1, 3, 4, 2)  # (batch_size, n_cameras, latent_height, latent_width, latent_depth)
        img_embeddings = (
            self.img_tokens_embedder(img_tokens)  # (batch_size, n_cameras, latent_height, latent_width, decoder_dim)
            + self.img_tokens_pos_embedding
        ).flatten(1, 3)  # (batch_size, n_cameras * latent_height * latent_width, decoder_dim)

        proprio_embedding = self.proprio_embedder(proprio)  # (batch_size, decoder_dim)
        z_embedding = self.z_embedder(z)  # (batch_size, decoder_dim)

        decoder_encoder_input = torch.cat(  # (batch_size, n_cameras * latent_height * latent_width + 2, decoder_dim)
            [
                img_embeddings,
                proprio_embedding.unsqueeze(1),
                z_embedding.unsqueeze(1)
            ],
            dim=1,
        )

        batch_size = z.size(0)
        decoder_decoder_input = self.decoder_decoder_pos_embedding.weight.expand(batch_size, -1, -1)

        decoder_output = self.decoder(decoder_encoder_input, decoder_decoder_input)
        chunk = self.action_projector(decoder_output)
        return chunk


class ACT(Module):
    def __init__(self, action_dim: int, chunk_len: int):
        super().__init__()
        self.z_dim = 32
        self.chunk_encoder = ACTEncoder(action_dim=action_dim, d_model=512, z_dim=self.z_dim, chunk_len=chunk_len)
        self.image_encoder = Sequential(
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            *list(
                resnet18(weights=ResNet18_Weights.IMAGENET1K_V1, norm_layer=FrozenBatchNorm2d).children()
            )[:-2]
        )
        self.chunk_decoder = ACTDecoder(action_dim=action_dim, d_model=512, z_dim=self.z_dim, chunk_len=chunk_len, img_tokens_shape=(3, 3, 512))

    def forward(
        self,
        img: Tensor,  # (batch_size, n_cameras, n_channels, height, width), float in [0, 1]
        proprio: Tensor,  # (batch_size, action_dim)
        chunk: Tensor | None,  # (batch_size, chunk_len, action_dim)
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        batch_size = proprio.size(0)
        if chunk is None:  # Inference
            z_mean, z_logvar = None, None
            z = torch.zeros(batch_size, self.z_dim).to(img.device)
        else:  # Training
            z_mean, z_logvar = self.chunk_encoder(proprio, chunk)
            eps = torch.randn_like(z_mean)
            z = z_mean + (z_logvar / 2).exp() * eps

        latent_imgs = self.image_encoder(img.flatten(0, 1)).unflatten(0, img.shape[:2])
        next_chunk = self.chunk_decoder(latent_imgs, proprio, z)
        return next_chunk, z_mean, z_logvar
