# %%
import torch
from scripts.act import ACT

# %%
batch_size = 2
n_cameras = 3
action_dim = 4
chunk_len = 5
img = torch.rand(batch_size, n_cameras, 3, 96, 96)  # Don't like that height and width are hardcoded in ACT
proprio = torch.randn(batch_size, action_dim)
chunk = torch.randn(batch_size, chunk_len, action_dim)

act = ACT(action_dim=action_dim, chunk_len=chunk_len)

# %%
# Checksum test (Expecting near 84M)
sum(p.numel() for p in ACT(action_dim=2, chunk_len=100).parameters())

# %%
# # Smoke tests
# CPU, training
chunk_pred, z_mean, z_logvar = act(img, proprio, chunk)
assert chunk_pred.shape == (batch_size, chunk_len, action_dim), "Wrong output dimension"
assert (chunk_pred[:, 0] - chunk_pred[:, 1]).abs().max() > 0, "Output chunks are constants"
assert z_mean is not None and z_logvar is not None, "z_mean or z_logvar missing"
# CPU, inference
chunk_pred, z_mean, z_logvar = act(img, proprio, chunk=None)
assert chunk_pred.shape == (batch_size, chunk_len, action_dim), "Wrong output dimension"
assert (chunk_pred[:, 0] - chunk_pred[:, 1]).abs().max() > 0, "Output chunks are constants"
# GPU, training
chunk_pred, z_mean, z_logvar = act.cuda()(img.cuda(), proprio.cuda(), chunk.cuda())
assert chunk_pred.is_cuda and z_mean.is_cuda and z_logvar.is_cuda
# GPU, inference
chunk_pred, z_mean, z_logvar = act.cuda()(img.cuda(), proprio.cuda(), chunk=None)
assert chunk_pred.is_cuda

# %%
# Gradient flow test
act = ACT(action_dim=action_dim, chunk_len=chunk_len)
act.zero_grad()
chunk_pred, z_mean, z_logvar = act(img, proprio, chunk)
loss = chunk_pred.abs().mean()
loss.backward()
for n, p in act.named_parameters():
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
loader = DataLoader(ds, batch_size=8, shuffle=False)
batch = next(iter(loader))
img = batch["observation.image"].unsqueeze(1).cuda()
proprio = batch["observation.state"].cuda()
proprio = (proprio - proprio.mean(axis=0)) / proprio.std(axis=0)
chunk = batch["action"].cuda()
chunk = (chunk - chunk.mean(axis=0)) / chunk.std(axis=0)

act = ACT(action_dim=2, chunk_len=chunk_len).cuda()
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
