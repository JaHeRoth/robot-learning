from functools import partial

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.optim import AdamW
from torch.utils.data import DataLoader

from scripts.act import ACT, ACTPolicy
from scripts.tasks import PUSHT, Task
from scripts.train_common import run_eval, train_loop


def act_loss(model, batch, stats, kl_weight):
    img = batch["observation.image"].unsqueeze(1).cuda()
    state_stats, action_stats = stats["observation.state"], stats["action"]
    proprio = (batch["observation.state"].cuda() - state_stats["mean"]) / state_stats["std"]
    chunk = (batch["action"].cuda() - action_stats["mean"]) / action_stats["std"]
    loss_mask = ~batch["action_is_pad"].cuda().unsqueeze(-1)

    chunk_pred, z_mean, z_logvar = model(img, proprio, chunk)
    l1_loss = ((chunk_pred - chunk).abs() * loss_mask).mean()
    kl_loss = -(1 + z_logvar - z_mean.pow(2) - z_logvar.exp()).sum(axis=1).mean() / 2
    return l1_loss + kl_weight * kl_loss


def train_act(
    task: Task, chunk_len: int = 100, n_action_steps: int = 16, seed: int = 0
):
    torch.manual_seed(seed)

    batch_size = 64
    lr = 2e-5
    weight_decay = 1e-4
    kl_weight = 10.0
    grad_clip_at = 10.0

    num_batches = 100_000
    eval_every = 10_000
    n_eval_envs = 50
    n_recorded = 10
    eval_start_seed = 1000

    fps = task.fps
    ds = LeRobotDataset(
        task.dataset_repo_id,
        delta_timestamps={"action": [i / fps for i in range(chunk_len)]}
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)

    model = ACT(action_dim=ds.meta.features["action"]["shape"][0], chunk_len=chunk_len).cuda()
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    stats = {
        obj: {
            attr: torch.as_tensor(
                ds.meta.stats[obj][attr], dtype=torch.float32, device="cuda"
            )
            for attr in ["mean", "std"]
        }
        for obj in ["action", "observation.state"]
    }

    env = task.make_env(n_eval_envs)
    eval_seeds = list(range(eval_start_seed, eval_start_seed + n_eval_envs))
    def eval_fn(model, step, out_dir):
        policy = ACTPolicy(model, dataset_stats=stats, n_action_steps=n_action_steps)
        return run_eval(
            env, policy, eval_seeds, step, out_dir=out_dir, record_n=n_recorded, fps=fps,
            imputed_reward=task.imputed_reward,
            image_key=task.image_key,
            state_key=task.state_key,
        )

    train_loop(
        model=model,
        loader=loader,
        opt=opt,
        loss_fn=partial(act_loss, kl_weight=kl_weight),
        num_batches=num_batches,
        out_dir="outputs/my_act",
        stats=stats,
        grad_clip_at=grad_clip_at,
        eval_every=eval_every,
        eval_fn=eval_fn,
    )


if __name__ == "__main__":
    train_act(task=PUSHT)
