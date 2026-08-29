# %%
import torch
from scripts.act import ACT

batch_size = 2
n_cameras = 3
action_dim = 4
chunk_len = 5
img = torch.rand(batch_size, n_cameras, 3, 96, 96)  # Don't like that height and width are hardcoded in ACT
proprio = torch.randn(batch_size, action_dim)
chunk = torch.randn(batch_size, chunk_len, action_dim)

act = ACT(action_dim=action_dim, chunk_len=chunk_len)

# %%
# Smoke tests
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
# Checksum test (Expecting near 84M)
sum(p.numel() for p in act.parameters())

# %%
