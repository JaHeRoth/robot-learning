"""jaheroth/so100_flip2_3cam_100: so100_flip2_100 recorded with a third stream,
observation.image_wrist_below, from the wrist camera mirrored to the other wrist-roll
branch: below the jaws at the grasp, where the real one is above. A bare <camera>, so
states and actions are so100_flip2_100's. One dataset, three policies: front only,
front + the real wrist camera, front + the mirrored one.

    python -m scripts.box.ablations.wristcam_3cam --out-dir outputs/wristcam_3cam_100
"""
from dataclasses import replace
from functools import partial
from pathlib import Path

import mujoco
from gymnasium.vector import AsyncVectorEnv
from gymnasium.wrappers import TimeLimit

import scripts.eval_so100_flip_wristcam as E
import scripts.generate_so100_flip_wristcam_data as G
from scripts.box.ablations.common import generate_cli
from scripts.box.ablations.wristcam_front import FRONT
from scripts.tasks import SO100_FLIP_WRISTCAM_100

SCENE = Path(__file__).resolve().parent / "scenes/wristcam_3cam/scene.xml"
CAMERAS = G.CAMERAS | {"observation.image_wrist_below": "wrist_below"}


def _make_one(horizon):
    return TimeLimit(E.SO100FlipWristcam(mujoco.MjModel.from_xml_path(str(SCENE))),
                     max_episode_steps=horizon)


def make_env(n_envs, horizon=E.HORIZON):
    # Each worker unpickles _make_one from this module, so it imports the patches too.
    return AsyncVectorEnv([partial(_make_one, horizon) for _ in range(n_envs)], context="spawn")


G.SCENE = SCENE
G.CAMERAS = CAMERAS
E.CAMERAS = CAMERAS
G.REPO_ID = "jaheroth/so100_flip2_3cam_100"

THREE_CAMERAS = replace(SO100_FLIP_WRISTCAM_100, dataset_repo_id=G.REPO_ID, make_env=make_env)
TASKS = {
    "so100_flip_wristcam_3cam_100_front": replace(THREE_CAMERAS, cameras=FRONT),
    "so100_flip_wristcam_3cam_100_wrist": replace(
        THREE_CAMERAS, cameras=FRONT | {"observation.image_wrist": "observation.image_wrist"}),
    "so100_flip_wristcam_3cam_100_wristbelow": replace(
        THREE_CAMERAS, cameras=FRONT | {"observation.image_wrist_below": "observation.image_wrist_below"}),
}

if __name__ == "__main__":
    generate_cli(G, n_episodes=100, randomize_speed=True)
