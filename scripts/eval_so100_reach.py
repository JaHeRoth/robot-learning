from pathlib import Path
from time import time
from typing import Iterable

import mujoco
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm
from gymnasium import Env, spaces
from mujoco import MjModel, MjData

from scripts.generate_so100_reach_data import sample_q_init_and_target, generate_expert_trajectory

REPO_ROOT = Path(__file__).resolve().parent.parent

class SO100Reach(Env):
    def __init__(self, seed: int, mjmodel: MjModel, mjdata: MjData):
        super().__init__()
        imgshape = (96, 96, 3)
        fps = 25

        self.rng = np.random.default_rng(seed)
        self.mjmodel = mjmodel
        self.mjdata = mjdata
        self.renderer = mujoco.Renderer(mjmodel, height=imgshape[0], width=imgshape[1])
        self.observation_space = spaces.Dict({
            "observation.image": spaces.Box(low=0, high=255, shape=imgshape, dtype=np.uint8),
            "observation.state": spaces.Box(
                low=mjmodel.jnt_range[:, 0], high=mjmodel.jnt_range[:, 1], dtype=np.float32
            ),
        })
        self.action_space = spaces.Box(
            low=mjmodel.actuator_ctrlrange[:, 0],
            high=mjmodel.actuator_ctrlrange[:, 1],
            dtype=np.float32,
        )
        self.n_substeps = round(1 / (fps * mjmodel.opt.timestep))
        geom_id = int(mjmodel.body("target").geomadr[0])
        self.distance_threshold = float(mjmodel.geom_size[geom_id][0])  # 0.02
        self.n_within_threshold = 10  # Matches linger on target in dataset
        self.n_within = 0

    def _capture_obs(self):
        mujoco.mj_forward(self.mjmodel, self.mjdata)
        self.renderer.update_scene(self.mjdata, camera="front")
        return {
            "observation.image": self.renderer.render(),
            "observation.state": self.mjdata.qpos.copy().astype(np.float32),
        }

    def reset(self, seed: int | None = None, options = None) -> tuple[dict, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.mjmodel, self.mjdata)
        q_init, self.q_target = sample_q_init_and_target(
            self.rng, self.mjmodel, self.mjdata, self.renderer
        )
        self.mjdata.qpos = q_init.copy()
        self.n_within = 0
        return self._capture_obs(), {}

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        self.mjdata.ctrl = action
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.mjmodel, self.mjdata)
        obs = self._capture_obs()

        tip = self.mjdata.site("tip").xpos
        goal = self.mjdata.body("target").xpos
        distance = float(np.linalg.norm(goal - tip))
        reward = -distance
        if distance <= self.distance_threshold:
            self.n_within += 1
        else:
            self.n_within = 0
        terminated = self.n_within >= self.n_within_threshold
        truncated = False  # TimeLimit wrapper owns this
        info = {"is_success": terminated}
        return obs, reward, terminated, truncated, info
