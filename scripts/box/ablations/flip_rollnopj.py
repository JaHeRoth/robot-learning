"""jaheroth/so100_flip_rollnopj_100: flip_roll and flip_nopj together, the pre-rolled
expert without drop-off noise.

    python -m scripts.box.ablations.flip_rollnopj --out-dir outputs/flip_rollnopj_100
"""
from dataclasses import replace

import scripts.box.ablations.flip_nopj  # noqa: F401  (its sample_episode)
import scripts.box.ablations.flip_roll  # noqa: F401  (its Expert)
import scripts.generate_so100_flip_data as G
from scripts.box.ablations.common import generate_cli
from scripts.tasks import SO100_FLIP

G.REPO_ID = "jaheroth/so100_flip_rollnopj_100"

TASKS = {"so100_flip_rollnopj_100": replace(SO100_FLIP, dataset_repo_id=G.REPO_ID)}

if __name__ == "__main__":
    generate_cli(G, n_episodes=100)
