"""jaheroth/so100_flip_nopj_100: the flip expert without its drop-off noise.

The flip expert aims the drop-off at SHELF_XY plus up to 15 mm of uniform noise per
axis, which the policy cannot see: label noise. Here it always aims at SHELF_XY. The
noise is the last draw, so every seed keeps its so100_flip_base100 scene.

    python -m scripts.box.ablations.flip_nopj --out-dir outputs/flip_nopj_100
"""
from dataclasses import replace

import numpy as np

import scripts.generate_so100_flip_data as G
from scripts.box.ablations.common import generate_cli
from scripts.generate_so100_flip_data import JAW, OPEN, SHELF_XY, SPAWN_HALF, SPAWN_MID
from scripts.tasks import SO100_FLIP


def sample_episode(rng, expert):
    """Glass pose, arm start and place spot. The spawn box is pre-vetted for
    reachability margin, so no rejection sampling is needed."""
    glass_xy = SPAWN_MID + rng.uniform(-SPAWN_HALF, SPAWN_HALF)
    q_init = np.clip(expert.home + rng.normal(0, 0.05, 6), expert.lb, expert.ub)
    q_init[JAW] = OPEN
    place_xy = SHELF_XY  # ablation: no drop-off noise
    return glass_xy, q_init, place_xy


G.sample_episode = sample_episode
G.REPO_ID = "jaheroth/so100_flip_nopj_100"

TASKS = {"so100_flip_nopj_100": replace(SO100_FLIP, dataset_repo_id=G.REPO_ID)}

if __name__ == "__main__":
    generate_cli(G, n_episodes=100)
