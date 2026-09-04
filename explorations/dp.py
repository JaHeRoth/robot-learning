# %%
import torch
from scripts.dp import DiffusionPolicy

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

dp = DiffusionPolicy(max_k=max_k, chunk_len=chunk_len)

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
# Checksum test
print(f"LeRobot's param count: {TODO}")
print(f"Our param count: {sum(p.numel() for p in dp.parameters())}")

# %%
# Gradient flow test
dp = DiffusionPolicy(max_k=max_k, chunk_len=chunk_len)
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
    delta_timestamps={"action": [i / fps for i in range(chunk_len)]}
)
loader = DataLoader(ds, batch_size=8, shuffle=True)
batch = next(iter(loader))
img = batch["observation.image"].unsqueeze(1).cuda()
proprio = batch["observation.state"].cuda()
proprio = (proprio - proprio.mean(axis=0)) / proprio.std(axis=0)
chunk = batch["action"].cuda()
chunk = (chunk - chunk.mean(axis=0)) / chunk.std(axis=0)

act = ACT(action_dim=2, chunk_len=chunk_len, cfg=ACTConfig(dropout=0.0)).cuda()
opt = AdamW(act.parameters(), lr=1e-4)
losses = []
for _ in tqdm(range(1000)):
    chunk_pred, z_mean, z_logvar = act(img, proprio, chunk)
    loss = F.l1_loss(chunk_pred, chunk)
    opt.zero_grad()
    loss.backward()
    opt.step()
    losses.append(loss.item())

plt.plot(losses)
plt.xlabel("Optimizer steps")
plt.ylabel("L1 loss")
plt.yscale("log")
plt.show()

# %%
