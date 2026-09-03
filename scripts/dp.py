import torch
from torch import Tensor
from torch.nn import Module, Linear, Sequential
from torchvision.models import resnet18


def _make_sequence_pos_embedding(step: Tensor, dim: int):
    assert dim % 2 == 0
    T = 10000
    w = 1 / T ** (2 * torch.arange(dim // 2) / dim)
    # TODO: Double-check that this works correctly when B>1
    return torch.stack([(step * w).sin(), (step * w).cos()]).permute(1, 0).flatten()


class UNet(Module):
    def forward(
        self,
        conditioner: Tensor,  # (B, n_obs * (n_resnet18_out_channels + dof) + dim_k_encoding)
        chunk: Tensor,  # (B, chunk_len, dof)
    ):
        raise NotImplementedError


class Denoiser(Module):
    def __init__(self, dim_k_encoding: int = 128):
        super().__init__()
        self.k_encoder = TODO
        self.unet = UNet()
    
    def forward(
        self,
        img_encoding: Tensor,  # (B, n_obs * n_resnet18_out_channels)
        proprio: Tensor,  # (B, n_obs, dof)
        step: Tensor,  # (B,)
        chunk: Tensor,  # (B, chunk_len, dof)
    ) -> Tensor:
        k_encoding = self.k_encoder(
            _make_sequence_pos_embedding(step=step, dim=self.k_encoder.in_features)  # TODO: Instead make part of self.k_encoder
        )
        conditioner = torch.cat(
            [img_encoding, proprio.flatten(start_dim=1), k_encoding],
            dim=-1,
        )
        eps_hat = self.unet(conditioner, chunk)
        return eps_hat


class DiffusionPolicy(Module):
    def __init__(self, chunk_len: int):
        super().__init__()
        self.chunk_len = chunk_len
        self.img_encoder = resnet18()  # TODO: Paper's modifications
        self.denoiser = Denoiser()

    def forward(
        self,
        imgs: Tensor,  # (B, n_obs, n_channels, height, width)
        proprio: Tensor,  # (B, n_obs, dof)
        step: int,
        chunk: Tensor,  # (B, chunk_len, dof)
    ) -> Tensor:
        img_encoding = (
            self.img_encoder(imgs.flatten(start_dim=0, end_dim=1))
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
                eps_hat = self.denoiser(img_encoding, proprio, step, chunk)
                chunk = TODO
        elif method == "ddim":
            raise NotImplementedError
        # TODO: Support Flow Matching?
        else:
            raise ValueError
        return chunk
