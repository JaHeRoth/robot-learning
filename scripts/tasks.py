"""Everything that differs between the environments the policies are trained on."""
from dataclasses import dataclass, replace
from typing import Callable

from gymnasium.vector import VectorEnv
from lerobot.envs.factory import make_env, make_env_config

from scripts.eval_so100_flip import make_so100_flip_env
from scripts.eval_so100_flip_wristcam import make_so100_flip_wristcam_env
from scripts.eval_so100_reach import make_so100_env


@dataclass(frozen=True)
class Task:
    dataset_repo_id: str
    make_env: Callable[[int], VectorEnv]  # n_envs -> vector env
    fps: int
    cameras: dict[str, str]  # dataset to env key map for cameras
    state_key: str
    imputed_reward: float  # Best achievable per-step reward, credited after success


PUSHT = Task(
    dataset_repo_id="lerobot/pusht",
    make_env=lambda n_envs: make_env(make_env_config("pusht"), n_envs=n_envs),
    fps=10,
    cameras={"observation.image": "pixels"},
    state_key="agent_pos",
    imputed_reward=0.95,  # Success threshold: 95% coverage of target T
)

SO100_REACH = Task(
    dataset_repo_id="jaheroth/so100_reach",
    make_env=lambda n_envs: make_so100_env(n_envs=n_envs, horizon=150),
    fps=25,
    cameras={"observation.image": "observation.image"},
    state_key="observation.state",
    imputed_reward=-0.02,  # Success threshold: tip inside the target cube
)

SO100_FLIP = Task(
    dataset_repo_id="jaheroth/so100_flip",
    make_env=lambda n_envs: make_so100_flip_env(n_envs=n_envs, horizon=300),
    fps=25,
    cameras={"observation.image": "observation.image"},
    state_key="observation.state",
    imputed_reward=-0.012,  # 0.012 is expert's mean distance from target on success
)

SO100_FLIP_100 = replace(SO100_FLIP, dataset_repo_id="jaheroth/so100_flip_100")

SO100_FLIP_WRISTCAM_100 = Task(
    dataset_repo_id="jaheroth/so100_flip2_100",
    make_env=lambda n_envs: make_so100_flip_wristcam_env(n_envs=n_envs, horizon=450),
    fps=25,
    cameras={
        "observation.image": "observation.image",
        "observation.image_wrist": "observation.image_wrist",
    },
    state_key="observation.state",
    imputed_reward=-0.002,  # 0.002 is expert's mean distance from target on success
)

SO100_FLIP_WRISTCAM_1K = replace(SO100_FLIP_WRISTCAM_100, dataset_repo_id="jaheroth/so100_flip2_1k")

TASKS = {
    "pusht": PUSHT,
    "so100_reach": SO100_REACH,
    "so100_flip": SO100_FLIP,
    "so100_flip_100": SO100_FLIP_100,
    "so100_flip_wristcam_100": SO100_FLIP_WRISTCAM_100,
    "so100_flip_wristcam_1k": SO100_FLIP_WRISTCAM_1K,
}
