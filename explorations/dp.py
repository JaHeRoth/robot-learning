# %%
import torch
from scripts.dp import DiffusionPolicy, DPConfig

from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionConditionalUnet1d
from lerobot.configs.types import FeatureType, PolicyFeature

# %%
batch_size = 4
n_obs = 2
proprio_dim = 6
max_k = 100
chunk_len = 48

imgs = torch.rand(batch_size, n_obs, 3, 96, 96)
proprio = torch.randn(batch_size, n_obs, proprio_dim)
k = torch.randint(low=1, high=max_k + 1, size=(batch_size,))
chunk = torch.randn(batch_size, chunk_len, proprio_dim)

dp_config = DPConfig(max_k=max_k, chunk_len=chunk_len)
dp = DiffusionPolicy(dp_config)

# %%
# Checksum test
reference_denoiser = DiffusionConditionalUnet1d(
    DiffusionConfig(
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))}
    ),
    global_cond_dim=dp_config.n_obs * (dp_config.latent_img_depth + dp_config.proprio_dim),
)
print(f"LeRobot's param count: {sum(p.numel() for p in reference_denoiser.parameters())}")
print(f"Our param count: {sum(p.numel() for p in dp.denoiser.parameters())}")

# %%
# # Smoke tests
# training
eps_hat = dp(imgs, proprio, k, chunk)
assert eps_hat.shape == chunk.shape, "Wrong output dimension"
# inference
ddpm_chunk = dp.sample(imgs, proprio, original=True)
assert ddpm_chunk.shape == chunk.shape, "Wrong output dimension"
ddim_chunk = dp.sample(imgs, proprio, original=False)
assert ddim_chunk.shape == chunk.shape, "Wrong output dimension"

# %%
# Gradient flow test
dp = DiffusionPolicy(DPConfig(max_k=max_k, chunk_len=chunk_len))
eps_hat = dp(imgs, proprio, k, chunk)
dp.zero_grad()
loss = eps_hat.abs().mean()
loss.backward()
for n, p in dp.named_parameters():
    assert p.grad.abs().max() > 0, n

# %%
# "Can overfit training data" test
import torch.nn.functional as F
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from matplotlib import pyplot as plt

fps = 10
ds = LeRobotDataset(
    "lerobot/pusht",
    delta_timestamps={
        "observation.image": [-1 / fps, 0],
        "observation.state": [-1 / fps, 0],
        "action": [i / fps for i in range(chunk_len)],
    },
)
torch.manual_seed(0)
B = 8
loader = DataLoader(ds, batch_size=B, shuffle=True)
batch = next(iter(loader))
imgs = batch["observation.image"].cuda()
proprio = batch["observation.state"].cuda()
proprio = (proprio - proprio.mean(axis=0)) / proprio.std(axis=0)
chunk = batch["action"].cuda()
chunk = (chunk - chunk.mean(axis=0)) / chunk.std(axis=0)

dp = DiffusionPolicy(DPConfig(proprio_dim=2)).cuda()
opt = AdamW(dp.parameters(), lr=1e-4)
noise = torch.randn_like(chunk)
k = torch.randint(low=1, high=101, size=(B,), device="cuda")
noised_chunk = (
    dp.alpha_bar[k].sqrt()[:, None, None] * chunk
    + (1 - dp.alpha_bar[k]).sqrt()[:, None, None] * noise
)
losses = []
for _ in tqdm(range(1000)):
    eps_hat = dp(imgs, proprio, k, chunk=noised_chunk)
    loss = F.mse_loss(eps_hat, noise)
    opt.zero_grad()
    loss.backward()
    opt.step()
    losses.append(loss.item())

plt.plot(losses)
plt.xlabel("Optimizer steps")
plt.ylabel("L2 loss")
plt.yscale("log")
plt.show()

# %%
