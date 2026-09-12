"""jaheroth/so100_flip_roll3cam_100: the pre-rolled expert (flip_roll) recorded with
flip_3cam's three streams. The pre-roll swaps the two wrist cameras' sides, so here
observation.image_wrist sits above the jaws at the grasp, as on the wristcam arm.
States and actions are so100_flip_roll100's.

    python -m scripts.box.ablations.flip_roll3cam --out-dir outputs/flip_roll3cam_100
"""
from dataclasses import replace

import scripts.box.ablations.flip_3cam as C
import scripts.box.ablations.flip_roll  # noqa: F401  (its Expert)
import scripts.generate_so100_flip_data as G
from scripts.box.ablations.common import generate_cli

G.REPO_ID = "jaheroth/so100_flip_roll3cam_100"

THREE_CAMERAS = replace(C.THREE_CAMERAS, dataset_repo_id=G.REPO_ID)
TASKS = {
    "so100_flip_roll3cam_100_wrist": replace(THREE_CAMERAS, cameras=C.WRIST),
    "so100_flip_roll3cam_100_wristbelow": replace(THREE_CAMERAS, cameras=C.WRIST_BELOW),
}

if __name__ == "__main__":
    generate_cli(G, n_episodes=100)
