"""Eval one checkpoint of the from-scratch ACT on 200 seeds at k=16; write a fragment."""
import sys

import polars as pl
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.factory import make_env, make_env_config
from lerobot.scripts.eval import rollout

from scripts.act import ACT
from scripts.my_act_policy import MyACTPolicy

step, off = sys.argv[1], int(sys.argv[2])
device = sys.argv[3] if len(sys.argv) > 3 else "cpu"
if device == "cpu":
    torch.set_num_threads(1)

stats = LeRobotDataset("lerobot/pusht").meta.stats
act = ACT(action_dim=2, chunk_len=100)
act.load_state_dict(torch.load(f"outputs/my_act/step_{step}.pt", map_location="cpu"))
policy = MyACTPolicy(act, stats).to(device)

n_envs = 200
env = make_env(make_env_config("pusht"), n_envs=n_envs)
horizon = env.call("_max_episode_steps")[0]
r = rollout(env, policy, seeds=list(range(off, off + n_envs)))
succeeded = r["success"].any(dim=1, keepdim=True)
mask = r["done"] & succeeded
imputed = r["reward"] * ~mask + 0.95 * mask
sums = imputed.sum(dim=1) + 0.95 * (horizon - imputed.shape[1])
pl.DataFrame([
    {"name": "my_act", "checkpoint": step, "n_action_steps": 16,
     "seed": off + i, "sum_imputed_reward": s, "success": u}
    for i, (s, u) in enumerate(zip(sums.tolist(), succeeded.squeeze(1).tolist()))
]).write_parquet(f"outputs/frags2/my_act_{step}_{off}.parquet")
print("FRAG DONE my_act", step, off)
