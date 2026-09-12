"""Front camera only, on the main wristcam datasets: the controls for the wristcam
ablations below, which all train front-only too. Tasks only; the data is the main
generator's.
"""
from dataclasses import replace

from scripts.tasks import SO100_FLIP_WRISTCAM_100, SO100_FLIP_WRISTCAM_1K

FRONT = {"observation.image": "observation.image"}
TASKS = {
    "so100_flip_wristcam_100_front": replace(SO100_FLIP_WRISTCAM_100, cameras=FRONT),
    "so100_flip_wristcam_1k_front": replace(SO100_FLIP_WRISTCAM_1K, cameras=FRONT),
}
