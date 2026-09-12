"""jaheroth/so100_flip_wrist_100: the flip task recorded with a wrist camera as well.

A bare <camera> on the fixed jaw at the wristcam scene's camera pose, with no mount
geometry, so the physics, and with it every state and action, is so100_flip_base100's.
Without the pre-roll this camera sits below the jaws at the grasp; on the wristcam arm
the same pose sits above them.

    python -m scripts.box.ablations.flip_wrist --out-dir outputs/flip_wrist_100
"""
import json
from dataclasses import replace
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from time import time

import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.vector import AsyncVectorEnv
from gymnasium.wrappers import TimeLimit
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

import scripts.generate_so100_flip_data as G
from scripts.box.ablations.common import generate_cli
from scripts.eval_so100_flip import SO100Flip
from scripts.generate_so100_flip_data import FPS, IMG, REPO_ROOT, TASK
from scripts.tasks import SO100_FLIP

SCENE = Path(__file__).resolve().parent / "scenes/flip_wrist/scene.xml"
CAMERAS = {"observation.image": "front", "observation.image_wrist": "wrist"}


class Cameras:
    """Expert mixin: record every camera in CAMERAS, not just the front one."""

    def capture(self):
        frame = {
            "observation.state": self.d.qpos[:6].copy().astype(np.float32),
            "action": self.d.ctrl[:6].copy().astype(np.float32),
        }
        for key, cam in CAMERAS.items():
            self.renderer.update_scene(self.d, camera=cam)
            frame[key] = self.renderer.render().copy()
        self.frames.append(frame)


def _init_worker():
    # ablation: mix the cameras into whichever expert is in place by now (flip_roll3cam)
    expert = type("Expert", (Cameras, G.Expert), {})
    G._worker["expert"] = expert(mujoco.MjModel.from_xml_path(str(G.SCENE)))


def generate(n_episodes=1000, seed=0, workers=None, out_dir=None):
    """Write n_episodes demonstrations to out_dir, resuming if it already holds some.

    Seeds are consumed in order and the next unused one is checkpointed after every
    episode, so an interrupted run picks up exactly where it stopped instead of
    regenerating from scratch. Pass disjoint `seed` ranges to shard across machines
    or across several processes on one machine.
    """
    root = Path(out_dir) if out_dir else REPO_ROOT / f"outputs/so100_flip_{int(time())}"
    progress = root / "progress.json"
    if progress.exists():
        state = json.loads(progress.read_text())
        dataset = LeRobotDataset(G.REPO_ID, root=root)
        n_ok, next_seed = dataset.num_episodes, state["next_seed"]
        print(f"resuming {root}: {n_ok} episodes done, next seed {next_seed}")
    else:
        model = mujoco.MjModel.from_xml_path(str(G.SCENE))
        joint_names = [model.joint(i).name for i in range(6)]
        dataset = LeRobotDataset.create(
            repo_id=G.REPO_ID,
            fps=FPS,
            features={
                # ablation: one video stream per camera
                **{key: {"dtype": "video", "shape": (*IMG, 3),
                         "names": ["height", "width", "channels"]} for key in CAMERAS},
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
    with Pool(workers, initializer=G._init_worker) as pool:
        with tqdm(total=n_episodes, initial=n_ok) as bar:
            # ordered imap, so "every seed below next_seed is done" stays true
            for offset, frames in enumerate(pool.imap(G._episode, seeds, chunksize=1)):
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


class SO100FlipCams(SO100Flip):
    """The flip eval env, observing every camera in `cameras`."""

    def __init__(self, mjmodel, cameras):
        super().__init__(mjmodel)
        self.cameras = cameras
        self.observation_space = spaces.Dict({
            **{key: spaces.Box(low=0, high=255, shape=(*IMG, 3), dtype=np.uint8)
               for key in cameras},
            "observation.state": self.observation_space["observation.state"],
        })

    def _capture_obs(self):
        mujoco.mj_forward(self.mjmodel, self.mjdata)
        obs = {}
        for key, cam in self.cameras.items():
            self.renderer.update_scene(self.mjdata, camera=cam)
            obs[key] = self.renderer.render()
        obs["observation.state"] = self.mjdata.qpos[:6].copy().astype(np.float32)
        return obs


def _make_one(scene, cameras, horizon):
    return TimeLimit(SO100FlipCams(mujoco.MjModel.from_xml_path(scene), cameras),
                     max_episode_steps=horizon)


def make_env(n_envs, scene=SCENE, cameras=CAMERAS, horizon=300):
    # spawn, not fork: a forked child inherits a broken GL context and deadlocks
    return AsyncVectorEnv([partial(_make_one, str(scene), cameras, horizon)
                           for _ in range(n_envs)], context="spawn")


G._init_worker = _init_worker
G.generate = generate
G.SCENE = SCENE
G.REPO_ID = "jaheroth/so100_flip_wrist_100"

FRONT = {"observation.image": "observation.image"}
WRIST = FRONT | {"observation.image_wrist": "observation.image_wrist"}
TASKS = {
    "so100_flip_wrist_100": replace(
        SO100_FLIP, dataset_repo_id=G.REPO_ID, make_env=make_env, cameras=WRIST),
    "so100_flip_wrist_100_front": replace(
        SO100_FLIP, dataset_repo_id=G.REPO_ID, make_env=make_env, cameras=FRONT),
}

if __name__ == "__main__":
    generate_cli(G, n_episodes=100)
