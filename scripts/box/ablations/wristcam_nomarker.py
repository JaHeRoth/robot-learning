"""jaheroth/so100_flip2_nomarker_100: so100_flip2_100 with the wrist camera and its mount
rendered invisible. In the front view they are a high-contrast blob riding next to the
gripper, a landmark the flip arm lacks. Only their alpha changes, so mass, inertia and
collision, and with them every state and action, are so100_flip2_100's. The eval env
uses the same scene.

    python -m scripts.box.ablations.wristcam_nomarker --out-dir outputs/wristcam_nomarker_100
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

SCENE = Path(__file__).resolve().parent / "scenes/wristcam_nomarker/scene.xml"


def _make_one(horizon):
    return TimeLimit(E.SO100FlipWristcam(mujoco.MjModel.from_xml_path(str(SCENE))),
                     max_episode_steps=horizon)


def make_env(n_envs, horizon=E.HORIZON):
    return AsyncVectorEnv([partial(_make_one, horizon) for _ in range(n_envs)], context="spawn")


G.SCENE = SCENE
G.REPO_ID = "jaheroth/so100_flip2_nomarker_100"

TASKS = {
    "so100_flip_wristcam_nomarker_100": replace(
        SO100_FLIP_WRISTCAM_100, dataset_repo_id=G.REPO_ID, make_env=make_env, cameras=FRONT),
}

if __name__ == "__main__":
    generate_cli(G, n_episodes=100, randomize_speed=True)
