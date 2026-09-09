"""Re-evaluate every checkpoint of a run over many seeds, with confidence intervals.

The in-training eval uses 50 episodes, which is a 95% CI of about +-0.14 around a
success rate of 0.4 -- too wide to rank checkpoints against each other. This runs
the same rollout over an arbitrary number of seeds and reports Wilson intervals.

    python -m scripts.box.sweep_checkpoints outputs/act_reach_1k --task so100_reach \
        --n-seeds 500 --n-envs 25 --device cpu
"""
import argparse
import csv
import math
import re
import time
from pathlib import Path

import torch

from scripts.act import ACT, ACTPolicy
from scripts.my_rollout import my_rollout
from scripts.tasks import TASKS


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def find(state: dict, suffix: str):
    for k, v in state.items():
        if k.endswith(suffix):
            return v
    raise KeyError(suffix)


def load_policy(ckpt_path: Path, image_keys, n_action_steps: int, device: str, use_ema: bool):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ck["ema_state"] if use_ema and "ema_state" in ck else ck["model_state"]
    # Shapes carry the architecture, so a checkpoint is self-describing.
    chunk_len = find(state, "decoder_decoder_pos_embedding.weight").shape[0]
    action_dim = find(state, "action_projector.weight").shape[0]
    model = ACT(action_dim=action_dim, chunk_len=chunk_len)
    model.load_state_dict(state)
    model.to(device).eval()
    stats = {k: {a: t.to(device).float() for a, t in v.items()} for k, v in ck["stats"].items()}
    return ACTPolicy(model, dataset_stats=stats, image_keys=image_keys,
                     n_action_steps=n_action_steps), chunk_len


def evaluate(env, policy, seeds, n_envs, imputed_reward, cameras, state_key, horizon):
    successes, sums = 0, []
    for i in range(0, len(seeds), n_envs):
        batch = seeds[i:i + n_envs]
        if len(batch) < n_envs:                 # keep the vector env full
            batch = batch + batch[: n_envs - len(batch)]
        out = my_rollout(env, policy, batch, record_n=0,
                         cameras=cameras, state_key=state_key)
        succeeded = out["success"].any(dim=1, keepdim=True)
        ended = out["done"].int().argmax(dim=1, keepdim=True)
        steps = torch.arange(out["reward"].shape[1])
        counts_real = (steps < ended) | ((steps == ended) & ~succeeded)
        sum_imputed = (out["reward"] * counts_real).sum(dim=1) + imputed_reward * (
            horizon - ended.squeeze(1)) * succeeded.squeeze(1)
        keep = len(seeds[i:i + n_envs])
        successes += int(succeeded.squeeze(1)[:keep].sum())
        sums += sum_imputed[:keep].tolist()
    return successes, sums


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("--task", required=True, choices=list(TASKS))
    p.add_argument("--n-seeds", type=int, default=500)
    p.add_argument("--n-envs", type=int, default=25)
    p.add_argument("--n-action-steps", type=int, default=16)
    p.add_argument("--start-seed", type=int, default=50_000, help="disjoint from training evals")
    p.add_argument("--device", default="cpu")
    p.add_argument("--ema", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    task = TASKS[args.task]
    ckpts = sorted(args.run_dir.glob("step_*.pt"),
                   key=lambda q: int(re.search(r"step_(\d+)", q.name).group(1)))
    if not ckpts:
        raise SystemExit(f"no step_*.pt in {args.run_dir}")
    print(f"{len(ckpts)} checkpoints, {args.n_seeds} seeds each, device={args.device}", flush=True)

    env = task.make_env(args.n_envs)
    horizon = env.call("_max_episode_steps")[0]
    seeds = list(range(args.start_seed, args.start_seed + args.n_seeds))
    out_path = args.out or args.run_dir / f"sweep_{args.n_seeds}seeds.csv"
    rows = []
    for ckpt in ckpts:
        step = int(re.search(r"step_(\d+)", ckpt.name).group(1))
        policy, chunk_len = load_policy(ckpt, task.cameras.keys(), args.n_action_steps,
                                        args.device, args.ema)
        t0 = time.time()
        with torch.no_grad():
            n_ok, sums = evaluate(env, policy, seeds, args.n_envs, task.imputed_reward,
                                  task.cameras, task.state_key, horizon)
        lo, hi = wilson(n_ok, len(seeds))
        mean_reward = sum(sums) / len(sums)
        rows.append(dict(step=step, n_episodes=len(seeds), n_success=n_ok,
                         success_rate=n_ok / len(seeds), ci_low=lo, ci_high=hi,
                         avg_sum_imputed_reward=mean_reward, chunk_len=chunk_len,
                         n_action_steps=args.n_action_steps))
        print(f"step {step:>7}: success={n_ok/len(seeds):.3f} "
              f"[{lo:.3f}, {hi:.3f}]  reward={mean_reward:.2f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    env.close()
    print(f"wrote {out_path}")
