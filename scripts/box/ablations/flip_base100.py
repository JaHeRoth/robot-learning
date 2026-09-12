"""jaheroth/so100_flip_base100: the flip expert unchanged, on seeds 0-99. The control
for the other flip ablations, which keep these seeds and so these scenes.

    python -m scripts.box.ablations.flip_base100 --out-dir outputs/flip_base100
"""
from dataclasses import replace

import scripts.generate_so100_flip_data as G
from scripts.box.ablations.common import generate_cli
from scripts.tasks import SO100_FLIP

G.REPO_ID = "jaheroth/so100_flip_base100"

TASKS = {"so100_flip_base100": replace(SO100_FLIP, dataset_repo_id=G.REPO_ID)}

if __name__ == "__main__":
    generate_cli(G, n_episodes=100)
