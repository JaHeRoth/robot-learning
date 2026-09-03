import torch
from torch import Tensor
from torch.nn import Embedding, Mish, Module, Linear, Sequential, Conv1d, Conv2d, Softmax, GroupNorm, ReLU
from torchvision.models import resnet18


def _make_sequence_pos_embedding(max_k: int, dim: int):
    assert dim % 2 == 0
    T = 10000
    w = 1 / T ** (2 * torch.arange(dim // 2) / dim)
    embeddings = [
        torch.stack([(i * w).sin(), (i * w).cos()]).permute(1, 0).flatten()
        for i in range(max_k)
    ]
    return torch.stack(embeddings)


class ResBlock(Module):
    def __init__(self):
        super().__init__()
        self.conv = Conv1d(in_channels=50, out_channels)


class UNet(Module):
    def forward(
        self,
        conditioner: Tensor,  # (B, n_obs * (n_resnet18_out_channels + dof) + dim_k_encoding)
        chunk: Tensor,  # (B, chunk_len, dof)
    ):
        chunk_img = chunk.permute(0, 2, 1)


class Denoiser(Module):
    def __init__(self, max_k: int, dim_k_encoding: int = 128):
        super().__init__()
        step_embedder = Embedding.from_pretrained(
            _make_sequence_pos_embedding(max_k=max_k, dim=dim_k_encoding)
        )
        self.step_encoder = Sequential(
            step_embedder, Linear(dim_k_encoding, 512), Mish(), Linear(512, dim_k_encoding)
        )
        self.unet = UNet()
    
    def forward(
        self,
        img_encoding: Tensor,  # (B, n_obs * n_resnet18_out_channels)
        proprio: Tensor,  # (B, n_obs, dof)
        step: Tensor,  # (B,)
        chunk: Tensor,  # (B, chunk_len, dof)
    ) -> Tensor:
        step_encoding = self.step_encoder(step)
        conditioner = torch.cat(
            [img_encoding, proprio.flatten(start_dim=1), step_encoding],
            dim=-1,
        )
        eps_hat = self.unet(conditioner, chunk)
        return eps_hat


class SpatialSoftmax(Module):
    def __init__(self):
        super().__init__()
        self.sm = Softmax()

    def forward(
        self,
        x: Tensor,  # (B * n_obs, trunk_channels, trunk_height, trunk_width)
        n_obs: int,
    ):
        trunk_height, trunk_width = x.shape[2:]
        pos_row = torch.arange(trunk_height).unsqueeze(1).expand(x.shape[2:])
        pos_col = torch.arange(trunk_width).unsqueeze(0).expand_as(pos_row)
        pos = torch.stack([pos_row, pos_col])  # (2, trunk_height, trunk_width)
        x_flat = x.flatten(-2).unsqueeze(2)  # (B * n_obs, trunk_channels, 1, trunk_height * trunk_width)
        pos_flat = pos.reshape(1, 1, 2, -1)  # (1, 1, 2, trunk_height * trunk_width)
        avg_pos = (self.sm(x_flat) * pos_flat).sum(-1)  # (B * n_obs, trunk_channels, 2)
        out = avg_pos.unflatten(0, (-1, n_obs)).flatten(1)  # (B, n_obs, trunk_channels * 2)
        if self.like_lerobot:
            out = out(self.lerobot_extra)
        return out


class DiffusionPolicy(Module):
    def __init__(self, chunk_len: int, like_lerobot: bool = True):
        super().__init__()
        self.chunk_len = chunk_len
        self.img_encoder = Sequential(
            *list(
                resnet18(
                    norm_layer=(
                        lambda num_channels: (
                            GroupNorm(num_groups=num_channels // 16, num_channels=num_channels)
                        )
                    )
                ).children()
            )[:-2],
            Conv2d(in_channels=512, out_channels=32, kernel_size=1),
            SpatialSoftmax(),
            *(
                [
                    Linear(in_features=64, out_features=64),
                    ReLU(),
                ]
                if like_lerobot
                else []
            )
        )
        self.denoiser = Denoiser()

    def forward(
        self,
        imgs: Tensor,  # (B, n_obs, n_channels, height, width)
        proprio: Tensor,  # (B, n_obs, dof)
        step: Tensor,  # (B,)
        chunk: Tensor,  # (B, chunk_len, dof)
    ) -> Tensor:
        img_encoding = (
            self.img_encoder(
                imgs.flatten(start_dim=0, end_dim=1)  # (B * n_obs, n_channels, height, width)
            )  # TODO
            .flatten(start_dim=1)
            .unflatten(dim=0, sizes=imgs.shape[:2])
        )
        eps_hat = self.denoiser(img_encoding, proprio, step, chunk)
        return eps_hat

    @torch.no_grad()
    def sample(
        self,
        imgs: Tensor,  # (B, n_obs, n_channels, height, width)
        proprio: Tensor,  # (B, n_obs, dof)
        method: str = "ddpm"  # TODO: Make enum
    ) -> Tensor:
        chunk = torch.randn(
            proprio.size(0), self.chunk_len, proprio.size(-1), device=imgs.device
        )
        img_encoding = self.img_encoder(imgs).flatten()
        if method == "ddpm":
            for step in reversed(range(1000)):  # TODO: Don't hardcode step count
                eps_hat = self.denoiser(img_encoding, proprio, torch.Tensor([step]), chunk)
                chunk = TODO
        elif method == "ddim":
            raise NotImplementedError
        # TODO: Support Flow Matching?
        else:
            raise ValueError
        return chunk
