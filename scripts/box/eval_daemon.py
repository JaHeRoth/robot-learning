"""Continuously scan for un-evaled (run, checkpoint) cells - including _200k
extensions' new checkpoints as training produces them - and eval each to 5000
seeds at k=16 via fragment workers. Rescans every cycle (no stale job lists);
exits when a full scan finds nothing to do and no training is running."""
import os
import subprocess
import time
from pathlib import Path

import polars as pl

PYBIN = str(Path.home() / "robot-learning/.pixi/envs/default/bin/python")
PARQ = "outputs/re_eval_per_episode.parquet"
POOL = 29
OFFSETS = list(range(0, 5000, 200))
EXT_CKPTS = ["200000", "180000", "160000", "140000", "120000"]
BASE_CKPTS = ["100000", "080000", "060000", "040000", "020000"]


def gpu_pool_size():
    try:
        used, total = map(int, subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"], text=True).split(", "))
        return 0  # CPU-only while running unattended: trainings own the GPU
    except Exception:
        return 0


def scan_jobs(have_df):
    jobs = []
    targets = []
    for run_dir in sorted(Path("outputs/train").iterdir()):
        if run_dir.name.startswith(("_", ".")) or not (run_dir / "checkpoints").is_dir():
            continue
        ckpts = sorted((d.name for d in (run_dir / "checkpoints").iterdir()
                        if d.name.isdigit()), reverse=True)
        inherited = {"_200k": 100000, "_500k": 200000, "_800k": 100000, "_2M": 800000}
        floor = next((v for k, v in inherited.items() if run_dir.name.endswith(k)), 0)
        targets += [(run_dir.name, c) for c in ckpts if int(c) > floor]
    # extensions first, newest checkpoints first
    targets.sort(key=lambda t: (not t[0].endswith(("_200k", "_500k", "_800k", "_2M")), -int(t[1])))
    for r, c in targets:
        have = have_df.filter((pl.col("name") == r) & (pl.col("checkpoint") == c))["seed"]
        for off in OFFSETS:
            if Path(f"outputs/frags2/{r}_{c}_{off}.parquet").exists():
                continue
            if len(have) and have.is_in(list(range(off, off + 200))).any():
                continue
            jobs.append((r, c, off))
    return jobs


active = []
spawned = set()
quiet = 0
while True:
    active = [(p, d) for p, d in active if p.poll() is None]
    have_df = pl.read_parquet(PARQ).filter(pl.col("n_action_steps") == 16)
    jobs = [j for j in scan_jobs(have_df) if j not in spawned]
    training = subprocess.run(["pgrep", "-fc", "lerobot-train"], capture_output=True).returncode == 0
    if not jobs and not active and not training:
        quiet += 1
        if quiet >= 30:  # 30 consecutive quiet minutes: queue handoff gaps are shorter
            print("DAEMON DONE: nothing left to eval and no training running", flush=True)
            break
    else:
        quiet = 0
    gpu_pool = gpu_pool_size()
    while jobs and len(active) < POOL:
        r, c, off = jobs.pop(0)
        spawned.add((r, c, off))
        n_gpu = sum(d == "cuda" for _, d in active)
        device = "cuda" if n_gpu < gpu_pool else "cpu"
        env = dict(os.environ)
        if device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        active.append((subprocess.Popen(
            [PYBIN, "scripts/eval_frag.py", r, c, str(off), device],
            stdout=subprocess.DEVNULL,
            stderr=open(f"outputs/frags2/err_{r}_{c}_{off}.txt", "w"),
            env=env), device))
        print("spawned", r, c, off, device, flush=True)
    time.sleep(60)
