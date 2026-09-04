import torch
from torch import Tensor
from torch.nn import Embedding, Mish, Module, Linear, Sequential, Conv1d, Conv2d, Softmax, GroupNorm, ReLU, ModuleList, ConvTranspose1d
from torchvision.models import resnet18
from torch.nn import functional as F


def _make_sequence_pos_embedding(length: int, dim: int):
    assert dim % 2 == 0
    T = 10000
    w = 1 / T ** (2 * torch.arange(dim // 2) / dim)
    embeddings = [
        torch.stack([(i * w).sin(), (i * w).cos()]).permute(1, 0).flatten()
        for i in range(length)
    ]
    return torch.stack(embeddings)


class ResBlock(Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = TODO


class UNet(Module):
    def __init__(self):
        super().__init__()
        self.down1 = ModuleList([
            ResBlock(in_channels=6, out_channels=512),
            ResBlock(in_channels=512, out_channels=512),
            Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1),
        ])
        self.down2 = ModuleList([
            ResBlock(in_channels=512, out_channels=1024),
            ResBlock(in_channels=1024, out_channels=1024),
            Conv1d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1),
        ])
        self.down3 = ModuleList([
            ResBlock(in_channels=1024, out_channels=2048),
            ResBlock(in_channels=2048, out_channels=2048),
        ])
        self.mid = ModuleList([
            ResBlock(in_channels=2048, out_channels=2048),
            ResBlock(in_channels=2048, out_channels=2048),
        ])
        self.up1 = ModuleList([
            ResBlock(in_channels=4096, out_channels=1024),
            ResBlock(in_channels=1024, out_channels=1024),
            ConvTranspose1d(in_channels=1024, out_channels=1024, kernel_size=4, stride=2, padding=1),
        ])
        self.up2 = ModuleList([
            ResBlock(in_channels=2048, out_channels=512),
            ResBlock(in_channels=512, out_channels=512),
            ConvTranspose1d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1),
        ])
        self.head = Sequential(
            Conv1d(in_channels=512, out_channels=512, kernel_size=5, padding=2),
            GroupNorm(num_groups=8, num_channels=512),
            Mish(),
            Conv1d(in_channels=512, out_channels=6, kernel_size=1),
            # TODO: Transpose between dims 1 and 2
        )

    def forward(
        self,
        conditioner: Tensor,  # (B, n_obs * (n_resnet18_out_channels + dof) + dim_k_encoding)
        chunk: Tensor,  # (B, chunk_len, dof)
    ):
        assert chunk.size(1) % 4 == 0, "Chunk length must be multiple of 4"
        chunk_img = chunk.permute(0, 2, 1)


class Denoiser(Module):
    def __init__(self, max_k: int, dim_k_encoding: int = 128):
        super().__init__()
        k_embedder = Embedding.from_pretrained(
            _make_sequence_pos_embedding(length=max_k + 1, dim=dim_k_encoding)
        )
        self.k_encoder = Sequential(
            k_embedder, Linear(dim_k_encoding, 512), Mish(), Linear(512, dim_k_encoding)
        )
        self.unet = UNet()
    
    def forward(
        self,
        img_encoding: Tensor,  # (B, n_obs * n_resnet18_out_channels)
        proprio: Tensor,  # (B, n_obs, dof)
        k: Tensor,  # (B,)
        chunk: Tensor,  # (B, chunk_len, dof)
    ) -> Tensor:
        k_encoding = self.k_encoder(k)
        conditioner = torch.cat(
            [img_encoding, proprio.flatten(start_dim=1), k_encoding],
            dim=-1,
        )
        eps_hat = self.unet(conditioner, chunk)
        return eps_hat


class SpatialSoftmax(Module):
    def __init__(self):
        super().__init__()
        self.softmax = Softmax(dim=-1)

    def forward(
        self,
        x: Tensor,  # (B * n_obs, conv_channels, trunk_height, trunk_width)
    ):
        trunk_height, trunk_width = x.shape[2:]
        pos_row = torch.arange(trunk_height).unsqueeze(1).expand(x.shape[2:]).to(x.device)  # (trunk_height, trunk_width)
        pos_col = torch.arange(trunk_width).unsqueeze(0).expand_as(pos_row).to(x.device)  # (trunk_height, trunk_width)
        pos = torch.stack([pos_row, pos_col])  # (2, trunk_height, trunk_width)
        x_flat = x.flatten(-2).unsqueeze(2)  # (B * n_obs, conv_channels, 1, trunk_height * trunk_width)
        pos_flat = pos.reshape(1, 1, 2, -1)  # (1, 1, 2, trunk_height * trunk_width)
        avg_pos = (self.softmax(x_flat) * pos_flat).sum(-1)  # (B * n_obs, conv_channels, 2)
        return avg_pos.flatten(-2)  # (B * n_obs, conv_channels * 2)


class ImgsEncoder(Module):
    def __init__(self, like_lerobot):
        super().__init__()
        self.like_lerobot = like_lerobot
        self.trunk = Sequential(
            *list(
                resnet18(
                    norm_layer=(
                        lambda num_channels: (
                            GroupNorm(num_groups=num_channels // 16, num_channels=num_channels)
                        )
                    )
                ).children()
            )[:-2]
        )
        self.conv = Conv2d(in_channels=512, out_channels=32, kernel_size=1)
        self.spatial_softmax = SpatialSoftmax()
        if like_lerobot:
            self.lerobot_extra = Sequential(
                Linear(in_features=64, out_features=64),
                ReLU(),
            )

    def forward(
        self,
        imgs: Tensor,  # (B, n_obs, n_channels, height, width)
    ) -> Tensor:
        trunk_in = imgs.flatten(start_dim=0, end_dim=1)  # (B * n_obs, n_channels, height, width)
        trunk_out = self.trunk(trunk_in)  # (B * n_obs, trunk_channels, trunk_height, trunk_width)
        conv_out = self.conv(trunk_out)  # (B * n_obs, conv_channels, trunk_height, trunk_width)
        ssm_out = self.spatial_softmax(conv_out)  # (B * n_obs, conv_channels * 2)
        if self.like_lerobot:
            ssm_out = self.lerobot_extra(ssm_out)
        return ssm_out.reshape(len(imgs), -1)  # (B, n_obs * conv_channels * 2)



class DiffusionPolicy(Module):
    def __init__(self, max_k: int, chunk_len: int, like_lerobot: bool = True):
        super().__init__()
        self.max_k = max_k
        self.chunk_len = chunk_len
        self.imgs_encoder = ImgsEncoder(like_lerobot=like_lerobot)
        self.denoiser = Denoiser(max_k=max_k)

        f = torch.cos(  # Squared cos gives constant angular velocity along quarter-circle from x_K to x_0
            (
                torch.arange(self.max_k + 1, dtype=torch.float32) / self.max_k + 0.008
            ) / 1.008 * torch.pi / 2
        ) ** 2
        alpha_bar_target = f / f[0]
        beta = (
            1 - alpha_bar_target / _prev(alpha_bar_target)
        ).clip(max=0.999)  # Clip to avoid degeneracy at edge
        # From here, all formulas follow from definition of q(x_k|x_{k-1}), that q(x_{k-1}|x_k,x_0)
        #  is Gaussian, and that q(x_{k-1}|x_k) is near-Gaussian for small beta_k.
        alpha = 1 - beta
        alpha_bar = alpha.cumprod(dim=0)  # Would have equaled alpha_bar_target if beta wasn't clipped
        beta_tilde = (1 - _prev(alpha_bar)) / (1 - alpha_bar) * beta
        beta_tilde[0] = 0.0  # Was 0/0, thus NaN, but is the lim of something going to 0
        # Buffers (unlike plain attributes) follow .to(device) and enter state_dict
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("beta_tilde", beta_tilde)

    def forward(
        self,
        imgs: Tensor,  # (B, n_obs, n_channels, height, width)
        proprio: Tensor,  # (B, n_obs, dof)
        k: Tensor,  # (B,)
        chunk: Tensor,  # (B, chunk_len, dof)
    ) -> Tensor:
        imgs_encoding = self.imgs_encoder(imgs)
        eps_hat = self.denoiser(imgs_encoding, proprio, k, chunk)
        return eps_hat

    @torch.no_grad()
    def sample(
        self,
        imgs: Tensor,  # (B, n_obs, n_channels, height, width)
        proprio: Tensor,  # (B, n_obs, dof)
        original: bool = False  # DDPM if true, DDIM if false
    ) -> Tensor:
        device = imgs.device
        chunk = torch.randn(
            proprio.size(0), self.chunk_len, proprio.size(-1), device=device
        )
        imgs_encoding = self.imgs_encoder(imgs)
        if original:
            z = torch.randn(self.max_k + 1, *chunk.size(), device=device)  # (max_k + 1, B, chunk_len, dof)
            for k in reversed(range(1, self.max_k + 1)):
                k_tensor = torch.full(
                    size=(len(imgs),), fill_value=k, dtype=torch.long, device=device
                )  # (B,)
                eps_hat = self.denoiser(imgs_encoding, proprio, k_tensor, chunk)
                chunk = (1 / self.alpha[k].sqrt()) * (chunk - self.beta[k] / (1 - self.alpha_bar[k]).sqrt() * eps_hat) + self.beta_tilde[k].sqrt() * z[k]
        else:
            k_step = 10
            for k in reversed(range(1, self.max_k + 1, k_step)):
                k_tensor = torch.full(
                    size=(len(imgs),), fill_value=k, dtype=torch.long, device=device
                )  # (B,)
                eps_hat = self.denoiser(imgs_encoding, proprio, k_tensor, chunk)
                target_k = max(0, k - k_step)
                chunk = (
                    (self.alpha_bar[target_k] / self.alpha_bar[k]).sqrt() * chunk
                    + (
                        (1 - self.alpha_bar[target_k]).sqrt()
                        - (self.alpha_bar[target_k] * (1 - self.alpha_bar[k]) / self.alpha_bar[k]).sqrt()
                    ) * eps_hat
                )
        return chunk


def _prev(x: Tensor, fill_val: float = 1.0) -> Tensor:
    """out[i] := x[i - 1] if i > 0 else fill_val"""
    return F.pad(x[:-1], (1, 0), value=fill_val)
