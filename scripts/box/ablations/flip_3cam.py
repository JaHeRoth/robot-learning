"""jaheroth/so100_flip_old3cam_100: flip_wrist plus a third stream,
observation.image_wrist_below, from the wrist camera mirrored to the other wrist-roll
branch. Without the pre-roll that one sits above the jaws at the grasp; its name comes
from the wristcam arm, where it sits below. States and actions are still
so100_flip_base100's.

    python -m scripts.box.ablations.flip_3cam --out-dir outputs/flip_3cam_100
"""
from dataclasses import replace
from functools import partial
from pathlib import Path

import scripts.box.ablations.flip_wrist as W
import scripts.generate_so100_flip_data as G
from scripts.box.ablations.common import generate_cli

SCENE = Path(__file__).resolve().parent / "scenes/flip_3cam/scene.xml"
CAMERAS = W.CAMERAS | {"observation.image_wrist_below": "wrist_below"}

W.CAMERAS = CAMERAS
G.SCENE = SCENE
G.REPO_ID = "jaheroth/so100_flip_old3cam_100"

THREE_CAMERAS = replace(W.TASKS["so100_flip_wrist_100"], dataset_repo_id=G.REPO_ID,
                        make_env=partial(W.make_env, scene=SCENE, cameras=CAMERAS))
WRIST = W.FRONT | {"observation.image_wrist": "observation.image_wrist"}
WRIST_BELOW = W.FRONT | {"observation.image_wrist_below": "observation.image_wrist_below"}
TASKS = {"so100_flip_old3cam_100_wristbelow": replace(THREE_CAMERAS, cameras=WRIST_BELOW)}

if __name__ == "__main__":
    generate_cli(G, n_episodes=100)
