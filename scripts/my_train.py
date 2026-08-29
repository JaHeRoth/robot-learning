import torch
from scripts.my_rollout import my_rollout
from scripts.act import ACT
import torch.nn.functional as F
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from matplotlib import pyplot as plt
import numpy as np
from lerobot.envs.factory import make_env, make_env_config
from gymnasium.vector import VectorEnv

def my_train():
    chunk_len = 100
    batch_size = 64
    lr = 2e-5
    weight_decay = 1e-4
    kl_weight = 10.0
    n_action_steps = 16

    num_batches = 100_000
    log_every = 100
    eval_every = 1000
    n_envs = 64

    torch.manual_seed(1000)
    fps = 10
    ds = LeRobotDataset(
        "lerobot/pusht",
        delta_timestamps={"action": [i / fps for i in range(chunk_len)]}
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    iter_loader = iter(loader)

    model = ACT(action_dim=ds.meta.features["action"]["shape"], chunk_len=chunk_len).cuda()
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    env = make_env(make_env_config("pusht"), n_envs=n_envs)
    horizon = env.call("_max_episode_steps")[0]

    losses = []
    avg_losses = []
    rolling_avg_loss = 0.0
    sum_imputeds = []
    success_rates = []
    step = 1
    while step <= num_batches:
        for batch in loader:
            img = batch["observation.image"].unsqueeze(1).cuda()
            proprio = batch["observation.state"].cuda()
            proprio = (proprio - proprio.mean(axis=0)) / proprio.std(axis=0)
            chunk = batch["action"].cuda()
            chunk = (chunk - chunk.mean(axis=0)) / chunk.std(axis=0)
            is_padding = batch["action_is_pad"].cuda()
            loss_mask = is_padding.not_().unsqueeze(-1)

            chunk_pred, z_mean, z_logvar = model(img, proprio, chunk)
            l1_loss = ((chunk_pred - chunk).abs() * loss_mask).mean()
            kl_loss = -(1 + z_logvar - z_mean.pow(2) - z_logvar.exp()).sum(axis=1).mean() / 2
            loss = l1_loss + kl_weight * kl_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            rolling_avg_loss += loss.item() / log_every

            if step % log_every == 0:
                print(f"Step {i + 1}/{num_batches}: Loss={rolling_avg_loss}")
                avg_losses.append(rolling_avg_loss)
                rolling_avg_loss = 0.0
            if step % eval_every == 0:
                model.eval()
                with torch.no_grad():
                    policy = TODO
                    seeds = list(range(n_envs))
                    result = my_rollout(env, policy, seeds)
                    reward, success, done = result["reward"], result["success"], result["done"]# Successful episodes contribute 0.95 from their success step through
                    # the horizon; failed episodes keep their real rewards throughout.
                    succeeded = success.any(dim=1, keepdim=True)
                    mask = done & succeeded
                    imputed_rewards = reward * ~mask + 0.95 * mask
                    sum_imputed = (
                        imputed_rewards.sum(dim=1)
                        + 0.95 * (horizon - imputed_rewards.shape[1])
                    )
                    sum_imputeds.append(sum_imputed)
                    avg_sum_imputed_rewards = sum_imputed.mean()
                    success_rate = succeeded.float().mean()
                    success_rates.append(success_rate)
                    print(f"{avg_sum_imputed_rewards=}, {success_rate=}")
                model.train()

