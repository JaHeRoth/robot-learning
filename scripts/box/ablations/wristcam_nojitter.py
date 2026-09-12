"""jaheroth/so100_flip2_nojitter_100: the wristcam expert without its speed and
trajectory jitter, on the same scenes as so100_flip2_100.

The main generator skips the jitter draws when jitter is off, which would shift every
later draw and so change the scenes. This copy always makes them, so only the motion
changes. The eval env samples scenes the same way, as it did on the box, so its seeds
are different scenes from the main wristcam env's.

    python -m scripts.box.ablations.wristcam_nojitter --out-dir outputs/wristcam_nojitter_100
"""
from dataclasses import replace
from functools import partial

import mujoco
import numpy as np
from gymnasium.vector import AsyncVectorEnv
from gymnasium.wrappers import TimeLimit

import scripts.eval_so100_flip_wristcam as E
import scripts.generate_so100_flip_wristcam_data as G
from scripts.box.ablations.common import generate_cli
from scripts.box.ablations.wristcam_front import FRONT
from scripts.generate_so100_flip_wristcam_data import (
    JAW, NOMINAL_SPEED, OPEN, SHELF_XY, SPAWN_HALF, SPAWN_MID, SPEED_RANGE,
)
from scripts.tasks import SO100_FLIP_WRISTCAM_100


def sample_episode(rng, expert, randomize_speed=False):
    """Glass pose, arm start and place spot. The spawn box is pre-vetted for
    reachability margin, so no rejection sampling is needed."""
    speed = rng.uniform(*SPEED_RANGE)  # ablation: drawn even when unused
    jitter = np.random.default_rng(rng.integers(2**32))  # ablation: drawn even when unused
    expert.speed = speed if randomize_speed else NOMINAL_SPEED
    expert.jitter = jitter if randomize_speed else None
    glass_xy = SPAWN_MID + rng.uniform(-SPAWN_HALF, SPAWN_HALF)
    q_init = np.clip(expert.home + rng.normal(0, 0.05, 6), expert.lb, expert.ub)
    q_init[JAW] = OPEN
    place_xy = SHELF_XY  # always the same spot: vary the motion, not the target
    return glass_xy, q_init, place_xy


def _make_one(horizon):
    return TimeLimit(E.SO100FlipWristcam(mujoco.MjModel.from_xml_path(str(G.SCENE))),
                     max_episode_steps=horizon)


def make_env(n_envs, horizon=E.HORIZON):
    # Each worker unpickles _make_one from this module, so it imports the patches too.
    return AsyncVectorEnv([partial(_make_one, horizon) for _ in range(n_envs)], context="spawn")


G.sample_episode = sample_episode
E.sample_episode = sample_episode
G.REPO_ID = "jaheroth/so100_flip2_nojitter_100"

TASKS = {
    "so100_flip_wristcam_nojitter_100": replace(
        SO100_FLIP_WRISTCAM_100, dataset_repo_id=G.REPO_ID, make_env=make_env, cameras=FRONT),
}

if __name__ == "__main__":
    generate_cli(G, n_episodes=100, randomize_speed=False)
