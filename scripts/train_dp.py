from pathlib import Path

import torch
# from scripts.my_rollout import my_rollout
from scripts.act import ACT
from torch.nn.utils import clip_grad_norm_
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from matplotlib import pyplot as plt
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from matplotlib import pyplot as plt
import torch
from scripts.dp import DiffusionPolicy, DPConfig

from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionConditionalUnet1d
from lerobot.configs.types import FeatureType, PolicyFeature

def my_train(seed: int | None = None):
    if seed is not None:
        torch.manual_seed(seed)
    
    chunk_len = 16
    batch_size = 64
    n_action_steps = 8
    ema_decay = 0.999

    lr = 1e-4
    weight_decay = 1e-6
    grad_clip_at = 10.0
    adam_betas = (0.95, 0.999)
    adam_warmup = 50

    num_batches = 100_000
    log_every = 100
    eval_every = 1000
    checkpoint_every = 20_000
    n_envs = 64

    Path("outputs/my_dp").mkdir(parents=True, exist_ok=True)

    fps = 10
    ds = LeRobotDataset(
        "lerobot/pusht",
        delta_timestamps={
            "observation.image": [-1 / fps, 0],
            "observation.state": [-1 / fps, 0],
            "action": [i / fps for i in range(chunk_len)],
        },
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)

    model = DiffusionPolicy(
        DPConfig(
            proprio_dim=ds.meta.features["action"]["shape"][0],
            chunk_len=chunk_len,
        )
    ).cuda()
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=adam_betas, n_warmup_steps=adam_warmup)

    # env = make_env(make_env_config("pusht"), n_envs=n_envs)
    # horizon = env.call("_max_episode_steps")[0]

    state_mean = torch.as_tensor(
        ds.meta.stats["observation.state"]["mean"], dtype=torch.float32, device="cuda"
    )
    state_std = torch.as_tensor(
        ds.meta.stats["observation.state"]["std"], dtype=torch.float32, device="cuda"
    )
    action_mean = torch.as_tensor(
        ds.meta.stats["action"]["mean"], dtype=torch.float32, device="cuda"
    )
    action_std = torch.as_tensor(
        ds.meta.stats["action"]["std"], dtype=torch.float32, device="cuda"
    )

    losses = []
    avg_losses = []
    rolling_avg_loss = 0.0
    avg_sum_imputed_rewards = []
    success_rates = []
    step = 1
    ema_sd = model.state_dict().deepcopy()
    while step <= num_batches:
        for batch in loader:
            if step > num_batches:
                break
            imgs = batch["observation.image"].cuda()
            proprio = batch["observation.state"].cuda()
            proprio = (proprio - state_mean) / state_std
            chunk = batch["action"].cuda()
            chunk = (chunk - action_mean) / action_std
            is_padding = batch["action_is_pad"].cuda()
            loss_mask = ~is_padding.unsqueeze(-1)

            noise = torch.randn_like(chunk)
            k = torch.randint(low=1, high=101, size=(B,), device="cuda")
            noised_chunk = (
                model.alpha_bar[k].sqrt()[:, None, None] * chunk
                + (1 - model.alpha_bar[k]).sqrt()[:, None, None] * noise
            )

            noise_pred = model(imgs, proprio, k, chunk=noised_chunk)
            loss = ((noise_pred - noise).pow(2) * loss_mask).mean()
            opt.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), grad_clip_at)
            opt.step()

            ema_sd = {
                n: ema_decay * w + (1 - ema_decay) * model.state_dict()[n]
                for n, w in ema_sd.items()
            }

            losses.append(loss.item())
            rolling_avg_loss += loss.item() / log_every

            if step % log_every == 0:
                print(f"Step {step}/{num_batches}: Loss={rolling_avg_loss}")
                avg_losses.append(rolling_avg_loss)
                rolling_avg_loss = 0.0
            # TODO: Uncomment this and write policy wrapper
            # if step % eval_every == 0:
            #     model.eval()
            #     with torch.no_grad():
            #         policy = TODO
            #         seeds = list(range(n_envs))
            #         result = my_rollout(env, policy, seeds)
            #         reward, success, done = result["reward"], result["success"], result["done"]# Successful episodes contribute 0.95 from their success step through
            #         # the horizon; failed episodes keep their real rewards throughout.
            #         succeeded = success.any(dim=1, keepdim=True)
            #         mask = done & succeeded
            #         imputed_rewards = reward * ~mask + 0.95 * mask
            #         sum_imputed = (
            #             imputed_rewards.sum(dim=1)
            #             + 0.95 * (horizon - imputed_rewards.shape[1])
            #         )
            #         avg_sum_imputed_reward = sum_imputed.mean()
            #         avg_sum_imputed_rewards.append(avg_sum_imputed_reward.item())
            #         success_rate = succeeded.float().mean()
            #         success_rates.append(success_rate.item())
            #         print(f"{avg_sum_imputed_reward=}, {success_rate=}")
            #     model.train()
            if step % checkpoint_every == 0:
                torch.save(model.state_dict(), f"outputs/my_act/step_{step:06d}.pt")
                torch.save(ema_sd, f"outputs/my_act/ema_step_{step:06d}.pt")
            step += 1
    np.save("outputs/my_act/losses.npy", losses)
    np.save("outputs/my_act/avg_losses.npy", avg_losses)
    np.save("outputs/my_act/avg_sum_imputed_rewards.npy", avg_sum_imputed_rewards)
    np.save("outputs/my_act/success_rates.npy", success_rates)

    plt.plot(range(1, num_batches + 1, log_every), avg_losses)
    plt.xlabel("Step")
    plt.ylabel("Training loss")
    plt.yscale("log")
    plt.grid()
    plt.savefig("outputs/my_act/loss_curve.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    my_train(seed=0)
