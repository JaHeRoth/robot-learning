"""Fragment-based mass evaluation, CPU-only workers, idempotent and cancelable.
Priority: top-family runs at ckpt 100k, then their 80k..20k, then remaining
policies at 100k by descending known mean."""
import os
import subprocess
import time
from pathlib import Path

import polars as pl

PYBIN = str(Path.home() / "robot-learning/.pixi/envs/default/bin/python")

def gpu_pool_size():
    try:
        used, total = map(int, subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"], text=True).split(", "))
        return max(0, int((total - used - 6000) / 650))
    except Exception:
        return 0
PARQ = "outputs/re_eval_per_episode.parquet"
POOL = 29
OFFSETS = list(range(0, 5000, 200))
CKPTS_DESC = ["100000", "080000", "060000", "040000", "020000"]
FAMILY = [
    "act_pusht_bs64_dec7", "act_pusht_bs64_dec7_seed1001", "act_pusht_bs64_dec7_seed1002",
    "act_pusht_bs64_chunk32_dec7", "act_pusht_bs64_chunk32_dec7_seed1001",
    "act_pusht_bs64_chunk32_dec7_seed1002",
]

runs = [p.name for p in Path("outputs/train").iterdir()
        if (p / "checkpoints/100000").exists()
        and not p.name.endswith("_200k") and not p.name.startswith("_")]
df = pl.read_parquet(PARQ).filter(pl.col("n_action_steps") == 16)
mean = {r: (df.filter((pl.col("name") == r) & (pl.col("checkpoint") == "100000"))
            ["sum_imputed_reward"].mean() or -1.0) for r in runs}

fam_here = [r for r in FAMILY if r in runs]
rest = sorted([r for r in runs if r not in FAMILY], key=lambda r: -mean[r])
# (run, ckpt) in priority order
targets = [(r, c) for c in CKPTS_DESC for r in fam_here
           if (Path("outputs/train") / r / "checkpoints" / c).exists()]
targets += [(r, "100000") for r in rest]

jobs = []
for r, c in targets:
    have = set(df.filter((pl.col("name") == r) & (pl.col("checkpoint") == c))["seed"].to_list())
    for off in OFFSETS:
        if Path(f"outputs/frags2/{r}_{c}_{off}.parquet").exists():
            continue
        if any(s in have for s in range(off, off + 200)):
            continue
        jobs.append((r, c, off))
print(f"{len(jobs)} chunks queued over {len(targets)} (run, ckpt) targets", flush=True)

GPU_POOL = gpu_pool_size()
print(f"GPU sub-pool: {GPU_POOL} of {POOL} workers", flush=True)

active = []  # (proc, device)
while jobs or active:
    active = [(p, d) for p, d in active if p.poll() is None]
    while jobs and len(active) < POOL:
        r, c, off = jobs.pop(0)
        n_gpu = sum(d == "cuda" for _, d in active)
        device = "cuda" if n_gpu < GPU_POOL else "cpu"
        env = dict(os.environ)
        if device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        active.append((subprocess.Popen(
            [PYBIN, "scripts/eval_frag.py", r, c, str(off), device],
            stdout=subprocess.DEVNULL,
            stderr=open(f"outputs/frags2/err_{r}_{c}_{off}.txt", "w"),
            env=env), device))
        print("spawned", r, c, off, device, flush=True)
    time.sleep(3)
print("MASS EVAL DONE", flush=True)
