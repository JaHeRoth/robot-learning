"""Watch outputs/my_act/ for checkpoints; eval each to 5000 seeds via CPU
fragment workers. Modest pool (shares cores with the main eval daemon).
Exits after 30 quiet minutes with my_train no longer running."""
import os
import re
import subprocess
import time
from pathlib import Path

PYBIN = str(Path.home() / "robot-learning/.pixi/envs/default/bin/python")
POOL = 10
OFFSETS = list(range(0, 5000, 200))

active = []
quiet = 0
while True:
    active = [p for p in active if p.poll() is None]
    steps = sorted(re.match(r"step_(\d+)\.pt", p.name).group(1)
                   for p in Path("outputs/my_act").glob("step_*.pt"))
    jobs = [(s, off) for s in reversed(steps) for off in OFFSETS
            if not Path(f"outputs/frags2/my_act_{s}_{off}.parquet").exists()]
    training = subprocess.run(["pgrep", "-fc", "scripts.my_train"], capture_output=True).returncode == 0
    if not jobs and not active and not training:
        quiet += 1
        if quiet >= 30:
            print("MY-ACT EVAL DONE", flush=True)
            break
    else:
        quiet = 0
    while jobs and len(active) < POOL:
        s, off = jobs.pop(0)
        active.append(subprocess.Popen(
            [PYBIN, "-m", "scripts.my_eval_frag", s, str(off), "cpu"],
            stdout=subprocess.DEVNULL,
            stderr=open(f"outputs/frags2/err_my_act_{s}_{off}.txt", "w"),
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}))
        print("spawned my_act", s, off, flush=True)
    time.sleep(60)
