"""Write one version's scripted-expert demonstrations to a LeRobotDataset.

Run through the version's module (sim.flip.flip or sim.flip.wristcam), which adds
its own options.
"""
import argparse
import json
import os
from multiprocessing import Pool, cpu_count
from pathlib import Path
from time import time

import mujoco
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

from sim.flip.expert import FPS, IMG, TASK, Version

REPO_ROOT = Path(__file__).resolve().parents[2]

_worker: dict = {}


def _init_worker(version):
    _worker["version"] = version
    _worker["expert"] = version.expert(
        mujoco.MjModel.from_xml_path(str(version.scene)), version.cameras)


def _episode(job):
    """One demonstration, or None if it did not end with the glass upright on the
    shelf and nothing but the pads touched. Judged on that outcome rather than on
    the intermediate IK checks: those are conservative and abort runs that go on to
    succeed, which would drop exactly the near-edge spawns the policy needs to see."""
    seed, sample_kwargs = job
    version, expert = _worker["version"], _worker["expert"]
    glass_xy, q_init, place_xy = version.sample_episode(
        np.random.default_rng(seed), expert, **sample_kwargs)
    if not expert.run(glass_xy, q_init, place_xy, strict=False):
        return None
    return expert.frames


def generate(version: Version, n_episodes=1000, seed=0, workers=None, out_dir=None,
             **sample_kwargs):
    """Write n_episodes demonstrations to out_dir, resuming if it already holds some.

    Seeds are consumed in order and the next unused one is checkpointed after every
    episode, so an interrupted run picks up exactly where it stopped instead of
    regenerating from scratch. Pass disjoint `seed` ranges to shard across machines
    or across several processes on one machine.
    """
    root = Path(out_dir) if out_dir else REPO_ROOT / f"outputs/{version.name}_{int(time())}"
    progress = root / "progress.json"
    if progress.exists():
        state = json.loads(progress.read_text())
        dataset = LeRobotDataset(version.repo_id, root=root)
        n_ok, next_seed = dataset.num_episodes, state["next_seed"]
        print(f"resuming {root}: {n_ok} episodes done, next seed {next_seed}")
    else:
        model = mujoco.MjModel.from_xml_path(str(version.scene))
        joint_names = [model.joint(i).name for i in range(6)]
        dataset = LeRobotDataset.create(
            repo_id=version.repo_id,
            fps=FPS,
            features={
                **{k: {"dtype": "video", "shape": (*IMG, 3),
                       "names": ["height", "width", "channels"]} for k in version.cameras},
                "observation.state": {"dtype": "float32", "shape": (6,), "names": joint_names},
                "action": {"dtype": "float32", "shape": (6,), "names": joint_names},
            },
            root=root,
        )
        n_ok, next_seed = 0, seed

    # Oversample: some fraction of draws fail the expert's checks and are dropped.
    workers = workers or max(1, min(cpu_count(), 32))
    remaining = n_episodes - n_ok
    if remaining <= 0:
        return dataset, n_ok
    seeds = range(next_seed, next_seed + int(remaining * 1.4) + 16)
    n_tried = 0
    with Pool(workers, initializer=_init_worker, initargs=(version,)) as pool:
        with tqdm(total=n_episodes, initial=n_ok) as bar:
            # ordered imap, so "every seed below next_seed is done" stays true
            jobs = ((s, sample_kwargs) for s in seeds)
            for offset, frames in enumerate(pool.imap(_episode, jobs, chunksize=1)):
                n_tried += 1
                if frames is not None:
                    for frame in frames:
                        dataset.add_frame(frame, task=TASK)
                    dataset.save_episode()
                    n_ok += 1
                    bar.update(1)
                progress.write_text(json.dumps({"next_seed": seeds[offset] + 1, "n_ok": n_ok}))
                if n_ok >= n_episodes:
                    break
    made = n_ok - (n_episodes - remaining)
    print(f"{n_ok} episodes total; this run made {made} from {n_tried} attempts "
          f"({100*made/max(n_tried,1):.0f}% clean)")
    return dataset, n_ok


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--out-dir", default=None, help="resumed in place if it already exists")
    p.add_argument("--seed", type=int, default=0, help="first seed; use disjoint ranges to shard")
    return p


def run(version: Version, args, **sample_kwargs):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    ds, n = generate(version, n_episodes=args.episodes, seed=args.seed,
                     workers=args.workers, out_dir=args.out_dir, **sample_kwargs)
    print(f"done: {n} episodes at {ds.root}")
