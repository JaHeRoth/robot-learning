import torch
from torch import Tensor
from torch.nn import Embedding, Mish, Module, Linear, Sequential, Conv1d, Conv2d, Softmax, GroupNorm, ReLU
from torchvision.models import resnet18


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
    def __init__(self, chain_len: int, dim_k_encoding: int = 128):
        super().__init__()
        step_embedder = Embedding.from_pretrained(
            _make_sequence_pos_embedding(length=chain_len, dim=dim_k_encoding)
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
    def __init__(self, chain_len: int, chunk_len: int, like_lerobot: bool = True):
        super().__init__()
        self.chain_len = chain_len
        self.chunk_len = chunk_len
        self.imgs_encoder = ImgsEncoder(like_lerobot=like_lerobot)
        self.denoiser = Denoiser(chain_len=chain_len)

    def forward(
        self,
        imgs: Tensor,  # (B, n_obs, n_channels, height, width)
        proprio: Tensor,  # (B, n_obs, dof)
        step: Tensor,  # (B,)
        chunk: Tensor,  # (B, chunk_len, dof)
    ) -> Tensor:
        imgs_encoding = self.imgs_encoder(imgs)
        eps_hat = self.denoiser(imgs_encoding, proprio, step, chunk)
        return eps_hat

    @torch.no_grad()
    def sample(
        self,
        imgs: Tensor,  # (B, n_obs, n_channels, height, width)
        proprio: Tensor,  # (B, n_obs, dof)
        original: bool = False  # DDPM if true, DDIM if false
    ) -> Tensor:
        chunk = torch.randn(
            proprio.size(0), self.chunk_len, proprio.size(-1), device=imgs.device
        )
        imgs_encoding = self.imgs_encoder(imgs)
        f = torch.cos(  # Cubed cos gives constant angular velocity along quarter-circle from x_K to x_0
            (
                torch.arange(self.chain_len + 1, dtype=torch.float32, device=imgs.device) / self.chain_len + 0.008
            ) / 1.008 * torch.pi / 2
        ) ** 2
        alpha_bar_target = f / f[0]
        beta = (
            1 - alpha_bar_target[1:] / alpha_bar_target[:1]
        ).clip(max=0.999)  # Clip to avoid degeneracy at edge
        # From here, all formulas follow from definition of q(x_k|x_{k-1}), that q(x_{k-1}|x_k,x_0)
        #  is Gaussian, and that q(x_{k-1}|x_k) is near-Gaussian for small beta_k.
        alpha = 1 - beta
        alpha_bar = alpha.cumprod()  # Would have equaled alpha_bar_target if beta wasn't clipped
        if original:
            beta_tilde = (1 - alpha_bar_target[:1]) / (1 - alpha_bar_target[1:]) * beta
            z = torch.rand(self.chain_len, *chunk.size())  # (chain_len, B, chunk_len, dof)
            for k in reversed(range(self.chain_len)):
                step_tensor = torch.full(
                    size=(len(imgs),), fill_value=k, dtype=torch.long, device=imgs.device
                )  # (B,)
                eps_hat = self.denoiser(imgs_encoding, proprio, step_tensor, chunk)
                chunk = (1 / alpha[k].sqrt()) * (chunk - beta[k] / (1 - alpha_bar[k]).sqrt() * eps_hat) + beta_tilde.sqrt() * z[k]
        else:
            step_size = 10
            for step in reversed(range(0, self.chain_len, step_size)):
                step_tensor = torch.full(
                    size=(len(imgs),), fill_value=step, dtype=torch.long, device=imgs.device
                )  # (B,)
                eps_hat = self.denoiser(imgs_encoding, proprio, step_tensor, chunk)
                chunk = (alpha_bar[step - step_size])
        return chunk
