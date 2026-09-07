"""Everything that differs between the environments the policies are trained on."""
from dataclasses import dataclass
from typing import Callable

from gymnasium.vector import VectorEnv
from lerobot.envs.factory import make_env, make_env_config

from scripts.eval_so100_reach import make_so100_env


@dataclass(frozen=True)
class Task:
    dataset_repo_id: str
    make_env: Callable[[int], VectorEnv]  # n_envs -> vector env
    fps: int
    image_key: str  # Env's key for the rendered frame
    state_key: str
    imputed_reward: float  # Best achievable per-step reward, credited after success


PUSHT = Task(
    dataset_repo_id="lerobot/pusht",
    make_env=lambda n_envs: make_env(make_env_config("pusht"), n_envs=n_envs),
    fps=10,
    image_key="pixels",
    state_key="agent_pos",
    imputed_reward=0.95,  # Success threshold: 95% coverage of target T
)

SO100_REACH = Task(
    dataset_repo_id="jaheroth/so100_reach",
    make_env=lambda n_envs: make_so100_env(n_envs=n_envs, horizon=150),
    fps=25,
    image_key="observation.image",
    state_key="observation.state",
    imputed_reward=-0.02,  # Success threshold: tip inside the target cube
)
