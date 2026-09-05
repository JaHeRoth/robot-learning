from pathlib import Path

import diffusers
import torch
from torch import Tensor
# from scripts.my_rollout import my_rollout
from torch.nn.utils import clip_grad_norm_
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from matplotlib import pyplot as plt
import numpy as np

from scripts.dp import DiffusionPolicy, DPConfig


def _normalize(x: Tensor, stats: dict) -> Tensor:
    return 2 * (x - stats["min"]) / (stats["max"] - stats["min"]) - 1


def _denormalize(x: Tensor, stats: dict) -> Tensor:
    return (x + 1) / 2 * (stats["max"] - stats["min"]) + stats["min"]


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
    adam_warmup = 500

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

    config = DPConfig(
        proprio_dim=ds.meta.features["observation.state"]["shape"][0],
        chunk_len=chunk_len,
    )
    model = DiffusionPolicy(config).cuda()
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=adam_betas)
    sched = diffusers.optimization.get_scheduler("cosine", opt, num_warmup_steps=adam_warmup, num_training_steps=num_batches)

    # env = make_env(make_env_config("pusht"), n_envs=n_envs)
    # horizon = env.call("_max_episode_steps")[0]

    stats = {
        obj: {
            attr: torch.as_tensor(
                ds.meta.stats[obj][attr], dtype=torch.float32, device="cuda"
            )
            for attr in ["min", "max", "mean", "std"]
        }
        for obj in ["action", "observation.state", "observation.image"]
    }

    losses = []
    avg_losses = []
    rolling_avg_loss = 0.0
    avg_sum_imputed_rewards = []
    success_rates = []
    step = 1
    ema_sd = {
        n: w.clone()
        for n, w in model.state_dict().items()
    }
    while step <= num_batches:
        for batch in loader:
            if step > num_batches:
                break
            imgs = batch["observation.image"].cuda()
            proprio = batch["observation.state"].cuda()
            proprio = _normalize(proprio, stats=stats["observation.state"])
            chunk = batch["action"].cuda()
            chunk = _normalize(chunk, stats=stats["action"])
            is_padding = batch["action_is_pad"].cuda()
            loss_mask = ~is_padding.unsqueeze(-1)

            noise = torch.randn_like(chunk)
            k = torch.randint(
                low=1, high=model.config.max_k + 1, size=(chunk.size(0),), device="cuda"
            )
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
            sched.step()

            model_sd = model.state_dict()
            with torch.no_grad():
                ema_sd = {
                    n: (
                        ema_decay * ema_sd[n]
                        + (1 - ema_decay) * model_sd[n]
                    )
                    for n in ema_sd.keys()
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
                torch.save(
                    {"model_config": config, "model_state": model.state_dict(), "ema_state": ema_sd, "stats": stats, "opt_state": opt.state_dict()},
                    f"outputs/my_dp/step_{step:06d}.pt",
                )
            step += 1
    np.save("outputs/my_dp/losses.npy", losses)
    np.save("outputs/my_dp/avg_losses.npy", avg_losses)
    np.save("outputs/my_dp/avg_sum_imputed_rewards.npy", avg_sum_imputed_rewards)
    np.save("outputs/my_dp/success_rates.npy", success_rates)

    plt.plot(range(1, num_batches + 1, log_every), avg_losses)
    plt.xlabel("Step")
    plt.ylabel("Training loss")
    plt.yscale("log")
    plt.grid()
    plt.savefig("outputs/my_dp/loss_curve.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    my_train(seed=0)
