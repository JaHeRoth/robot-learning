"""Eval one (policy, checkpoint) on 200 seeds at k=16; write a parquet fragment."""
import sys
import torch
import polars as pl
from lerobot.envs.factory import make_env, make_env_config
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.scripts.eval import rollout

name, ckpt, off = sys.argv[1], sys.argv[2], int(sys.argv[3])
device = sys.argv[4] if len(sys.argv) > 4 else "cuda"
if device == "cpu":
    torch.set_num_threads(1)
n_envs = 200
env = make_env(make_env_config("pusht"), n_envs=n_envs)
horizon = env.call("_max_episode_steps")[0]
policy = ACTPolicy.from_pretrained(f"outputs/train/{name}/checkpoints/{ckpt}/pretrained_model").to(device)
policy.config.n_action_steps = 16
r = rollout(env, policy, seeds=list(range(off, off + n_envs)))
succeeded = r["success"].any(dim=1, keepdim=True)
mask = r["done"] & succeeded
imputed = r["reward"] * ~mask + 0.95 * mask
sums = imputed.sum(dim=1) + 0.95 * (horizon - imputed.shape[1])
pl.DataFrame([
    {"name": name, "checkpoint": ckpt, "n_action_steps": 16,
     "seed": off + i, "sum_imputed_reward": s, "success": u}
    for i, (s, u) in enumerate(zip(sums.tolist(), succeeded.squeeze(1).tolist()))
]).write_parquet(f"outputs/frags2/{name}_{ckpt}_{off}.parquet")
print("FRAG DONE", name, ckpt, off)
