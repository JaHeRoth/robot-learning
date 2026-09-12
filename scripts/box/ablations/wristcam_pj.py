"""jaheroth/so100_flip2_pj_100: wristcam_nojitter plus the flip expert's drop-off noise,
up to 15 mm per axis around SHELF_XY. The noise is drawn last, so every seed keeps its
wristcam_nojitter scene, and the eval env (which draws no noise) is that one too.

99 episodes, not 100: the generated set's episode 69 (seed 73) had a corrupt wrist
video and was cut before upload, so this skips that seed.

    python -m scripts.box.ablations.wristcam_pj --out-dir outputs/wristcam_pj_99
"""
from dataclasses import replace

import numpy as np

import scripts.box.ablations.wristcam_nojitter as N
import scripts.generate_so100_flip_wristcam_data as G
from scripts.box.ablations.common import generate_cli
from scripts.generate_so100_flip_wristcam_data import SHELF_XY

PLACE_JITTER = 0.015
DROPPED_SEED = 73


def sample_episode(rng, expert, randomize_speed=False):
    glass_xy, q_init, _ = N.sample_episode(rng, expert, randomize_speed)
    return glass_xy, q_init, SHELF_XY + rng.uniform(-PLACE_JITTER, PLACE_JITTER, 2)


def _episode(args):
    """One demonstration, or None if it did not end with the glass upright on the
    shelf and nothing but the pads touched. Judged on that outcome rather than on
    the intermediate IK checks: those are conservative and abort runs that go on to
    succeed, which would drop exactly the near-edge spawns the policy needs to see."""
    seed, randomize_speed = args
    if seed == DROPPED_SEED:  # ablation
        return None
    expert = G._worker["expert"]
    glass_xy, q_init, place_xy = sample_episode(
        np.random.default_rng(seed), expert, randomize_speed)
    if not expert.run(glass_xy, q_init, place_xy, strict=False):
        return None
    return expert.frames


G.sample_episode = sample_episode
G._episode = _episode
G.REPO_ID = "jaheroth/so100_flip2_pj_100"

TASKS = {
    "so100_flip_wristcam_pj_100": replace(
        N.TASKS["so100_flip_wristcam_nojitter_100"], dataset_repo_id=G.REPO_ID),
}

if __name__ == "__main__":
    generate_cli(G, n_episodes=99, randomize_speed=False)
