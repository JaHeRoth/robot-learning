from enum import Enum
from functools import partial

import diffusers
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader

from lerobot.envs.factory import make_env, make_env_config

from scripts.dp import (
    DiffusionModel,
    GenConfig,
    GenPolicy,
    FlowMatchingModel,
    denormalize,
    normalize,
)
from scripts.train_common import run_eval, train_loop


def _normalize(x: Tensor, stats: dict) -> Tensor:
    return normalize(x, min=stats["min"], max=stats["max"])


def _denormalize(x: Tensor, stats: dict) -> Tensor:
    return denormalize(x, min=stats["min"], max=stats["max"])


def random_crop(
    imgs: Tensor,  # (B, n_obs, n_channels, height, width)
    crop: int,
    full = 96,  # Assuming squared images
):
    start_row = torch.randint(low=0, high=full - crop + 1, size=imgs.shape[0:1])
    start_col = torch.randint(low=0, high=full - crop + 1, size=imgs.shape[0:1])
    return torch.stack(
        [
            img[:, :, y:y + crop, x:x + crop]
            for img, y, x in zip(imgs, start_row, start_col)
        ]
    )


def dp_loss(model, batch, stats, crop):
    imgs = random_crop(batch["observation.image"], crop=crop).cuda()
    proprios = _normalize(batch["observation.state"].cuda(), stats=stats["observation.state"])
    chunk = _normalize(batch["action"].cuda(), stats=stats["action"])
    loss_mask = ~batch["action_is_pad"].cuda().unsqueeze(-1)

    noise = torch.randn_like(chunk)
    k = torch.randint(
        low=1, high=model.config.max_k + 1, size=chunk.shape[0:1]
    ).cuda()
    noised_chunk = (
        model.alpha_bar[k].sqrt()[:, None, None] * chunk
        + (1 - model.alpha_bar[k]).sqrt()[:, None, None] * noise
    )
    noise_pred = model(imgs, proprios, k, chunk=noised_chunk)
    return ((noise_pred - noise).pow(2) * loss_mask).mean()


def fmp_loss(model, batch, stats, crop):
    imgs = random_crop(batch["observation.image"], crop=crop).cuda()
    proprios = _normalize(batch["observation.state"].cuda(), stats=stats["observation.state"])
    chunk = _normalize(batch["action"].cuda(), stats=stats["action"])
    loss_mask = ~batch["action_is_pad"].cuda().unsqueeze(-1)

    noise = torch.randn_like(chunk)
    velocity = noise - chunk
    t = torch.rand(size=chunk.shape[0:1]).cuda()
    noised_chunk = (1 - t[:, None, None]) * chunk + t[:, None, None] * noise
    velocity_pred = model(imgs, proprios, t, chunk=noised_chunk)
    return ((velocity_pred - velocity).pow(2) * loss_mask).mean()


class LossType(Enum):
    DIFFUSION = "diffusion"
    FLOW_MATCHING = "flow_matching"


def train_dp(loss_type: LossType, seed: int = 0):
    torch.manual_seed(seed)

    chunk_len = 16
    batch_size = 64
    ema_decay = 0.999

    lr = 1e-4
    weight_decay = 1e-6
    grad_clip_at = 10.0
    adam_betas = (0.95, 0.999)
    adam_warmup = 500

    num_batches = 100_000
    drop_n_last_frames = 7
    crop = 84

    n_action_steps = 8
    n_steps = 10  # DDIM steps for DP, Euler steps for FMP
    eval_every = 10_000
    n_eval_envs = 50
    n_recorded = 10
    eval_start_seed = 1000
    imputed_reward = 0.95  # PushT's max per-step reward
    image_key, state_key = "pixels", "agent_pos"

    fps = 10
    ds = LeRobotDataset(
        "lerobot/pusht",
        delta_timestamps={
            "observation.image": [-1 / fps, 0],
            "observation.state": [-1 / fps, 0],
            "action": [i / fps for i in range(chunk_len)],
        },
    )
    sampler = EpisodeAwareSampler(
        episode_data_index=ds.episode_data_index, drop_n_last_frames=drop_n_last_frames, shuffle=True
    )
    loader = DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=4)

    config = GenConfig(
        proprio_dim=ds.meta.features["observation.state"]["shape"][0],
        chunk_len=chunk_len,
    )
    if loss_type == LossType.DIFFUSION:
        loss_fn = dp_loss
        out_dir = "outputs/my_dp"
        model = DiffusionModel(config).cuda()
    else:
        loss_fn = fmp_loss
        out_dir = "outputs/my_fmp"
        model = FlowMatchingModel(config).cuda()

    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=adam_betas)
    sched = diffusers.optimization.get_scheduler("cosine", opt, num_warmup_steps=adam_warmup, num_training_steps=num_batches)

    stats = {
        obj: {
            attr: torch.as_tensor(
                ds.meta.stats[obj][attr], dtype=torch.float32, device="cuda"
            )
            for attr in ["min", "max", "mean", "std"]
        }
        for obj in ["action", "observation.state", "observation.image"]
    }

    env = make_env(make_env_config("pusht"), n_envs=n_eval_envs)
    eval_seeds = list(range(eval_start_seed, eval_start_seed + n_eval_envs))
    def eval_fn(model, step, out_dir):
        policy = GenPolicy(
            model,
            dataset_stats=stats,
            n_action_steps=n_action_steps,
            crop=crop,
            n_steps=n_steps,
        )
        return run_eval(
            env, policy, eval_seeds, step, out_dir=out_dir, record_n=n_recorded, fps=fps,
            imputed_reward=imputed_reward,
            image_key=image_key,
            state_key=state_key,
        )

    train_loop(
        model=model,
        loader=loader,
        opt=opt,
        sched=sched,
        loss_fn=partial(loss_fn, crop=crop),
        stats=stats,
        num_batches=num_batches,
        out_dir=out_dir,
        ema_decay=ema_decay,
        grad_clip_at=grad_clip_at,
        checkpoint_extra={"model_config": config},
        eval_every=eval_every,
        eval_fn=eval_fn,
    )


if __name__ == "__main__":
    train_dp(loss_type=LossType.FLOW_MATCHING)
