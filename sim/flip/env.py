"""Rollout environment for the SO-100 flip task, either version.

Success is the glass standing upright on the shelf *and released* -- holding it in
place at the goal does not count, since the demonstrations always let go.

    python -m sim.flip.env wristcam   # replay the expert through the env
"""
import argparse
from typing import Iterable

import mujoco
import numpy as np
from gymnasium import Env, spaces
from gymnasium.vector import AsyncVectorEnv
from gymnasium.wrappers import TimeLimit
from mujoco import MjModel

from sim.flip.expert import FPS, IMG, SHELF_TOP, SHELF_XY, Version, servo_ctrl

POS_TOLERANCE = 0.05   # of the glass origin from the target spot on the shelf
UPRIGHT = 0.9          # cos of the glass axis against +z
# Frames the glass must stay upright, on target and untouched before it counts.
# 2 s at 25 fps, long enough to cover the whole retreat: releasing cleanly and
# backing away without dragging the glass off the shelf is part of the task, and
# a short window would end the episode before the arm had done it.
N_WITHIN = 50


class SO100Flip(Env):
    def __init__(self, mjmodel: MjModel, version: Version):
        super().__init__()
        self.mjmodel = mjmodel
        self.mjdata = mujoco.MjData(mjmodel)
        self.scratch = mujoco.MjData(mjmodel)
        self.renderer = mujoco.Renderer(mjmodel, height=IMG[0], width=IMG[1])
        # Maps the dataset's key -> the camera in the scene, so the observation the
        # policy sees at eval is built exactly as the data was.
        self.cameras = version.cameras
        self.sample_episode = version.sample_episode
        self.observation_space = spaces.Dict({
            **{k: spaces.Box(low=0, high=255, shape=(*IMG, 3), dtype=np.uint8)
               for k in self.cameras},
            "observation.state": spaces.Box(
                low=mjmodel.jnt_range[:6, 0], high=mjmodel.jnt_range[:6, 1], dtype=np.float32
            ),
        })
        self.action_space = spaces.Box(
            low=mjmodel.actuator_ctrlrange[:, 0],
            high=mjmodel.actuator_ctrlrange[:, 1],
            dtype=np.float32,
        )
        self.n_substeps = round(1 / (FPS * mjmodel.opt.timestep))
        self.key = [mjmodel.key(i).name for i in range(mjmodel.nkey)].index("start")
        self.kp = float(mjmodel.actuator_gainprm[0, 0])
        # sample_episode reads these off the expert; the env stands in for it.
        self.home = mjmodel.key("start").qpos[:6].copy()
        self.lb, self.ub = mjmodel.jnt_range[:6, 0].copy(), mjmodel.jnt_range[:6, 1].copy()
        self.pads = {mjmodel.geom(i).id for i in range(mjmodel.ngeom)
                     if "jaw_pad" in mjmodel.geom(i).name}
        self.glass_geoms = {mjmodel.geom(i).id for i in range(mjmodel.ngeom)
                            if mjmodel.geom(i).name.startswith("glass")}
        self.goal = np.r_[SHELF_XY, SHELF_TOP]
        self.n_within = 0
        # A fresh renderer's first frame of the scene is off by a pixel from every
        # later one, which would make the first rollout after creating the env
        # irreproducible. Render a throwaway one.
        self._capture_obs()

    def _capture_obs(self):
        mujoco.mj_forward(self.mjmodel, self.mjdata)
        obs = {}
        for key, cam in self.cameras.items():
            self.renderer.update_scene(self.mjdata, camera=cam)
            obs[key] = self.renderer.render()
        obs["observation.state"] = self.mjdata.qpos[:6].copy().astype(np.float32)
        return obs

    def _held(self):
        """Is the gripper still touching the glass?"""
        for c in self.mjdata.contact[:self.mjdata.ncon]:
            pair = {int(c.geom1), int(c.geom2)}
            if pair & self.pads and pair & self.glass_geoms:
                return True
        return False

    def reset(self, *, seed: int | None = None, options=None) -> tuple[dict, dict]:
        super().reset(seed=seed)
        glass_xy, q_init, _ = self.sample_episode(self.np_random, self)
        mujoco.mj_resetDataKeyframe(self.mjmodel, self.mjdata, self.key)
        self.mjdata.qpos[:6] = q_init
        self.mjdata.qpos[6:8] = glass_xy
        mujoco.mj_forward(self.mjmodel, self.mjdata)
        # Hold the start pose against gravity and let the glass settle, exactly as
        # the demonstrations do, so the first observation is on-distribution.
        self.mjdata.ctrl[:6] = servo_ctrl(
            self.mjmodel, self.scratch, self.mjdata.qpos, q_init, self.kp
        )
        for _ in range(100):
            mujoco.mj_step(self.mjmodel, self.mjdata)
        self.n_within = 0
        return self._capture_obs(), {}

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        self.mjdata.ctrl = action
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.mjmodel, self.mjdata)
        obs = self._capture_obs()

        glass = self.mjdata.body("glass")
        distance = float(np.linalg.norm(glass.xpos - self.goal))
        tilt = float(glass.xmat.reshape(3, 3)[2, 2])
        reward = -distance
        placed = distance <= POS_TOLERANCE and tilt >= UPRIGHT and not self._held()
        self.n_within = self.n_within + 1 if placed else 0
        terminated = self.n_within >= N_WITHIN
        return obs, reward, terminated, False, {"is_success": terminated}


def make_env(version: Version, n_envs: int, horizon: int | None = None) -> AsyncVectorEnv:
    horizon = horizon or version.horizon

    def thunk():
        mjmodel = mujoco.MjModel.from_xml_path(str(version.scene))
        return TimeLimit(SO100Flip(mjmodel, version), max_episode_steps=horizon)
    # spawn, not fork: a forked child inherits a broken GL context and deadlocks
    return AsyncVectorEnv([thunk for _ in range(n_envs)], context="spawn")


def eval_experts(version: Version, seeds: Iterable[int], horizon: int | None = None) -> tuple[float, float]:
    """Replay the scripted expert through the env to measure the achievable ceiling.

    Also a consistency check: the env and the generator must agree on the reset and
    the physics, or the recorded actions will not reproduce the demonstration.
    """
    horizon = horizon or version.horizon
    mjmodel = mujoco.MjModel.from_xml_path(str(version.scene))
    env = SO100Flip(mjmodel, version)
    expert = version.expert(mjmodel, version.cameras)
    seeds = list(seeds)
    sum_reward = success = 0.0
    at_success = []
    for seed in seeds:
        glass_xy, q_init, place_xy = version.sample_episode(np.random.default_rng(seed), expert)
        if not expert.run(glass_xy, q_init, place_xy, strict=False):
            continue
        actions = [f["action"] for f in expert.frames]
        env.reset(seed=seed)
        total = 0.0
        for i in range(horizon):
            _, reward, terminated, _, _ = env.step(actions[min(i, len(actions) - 1)])
            if terminated:
                at_success.append(-reward)
                total += reward * (horizon - i)   # imputed from here on
                success += 1 / len(seeds)
                break
            total += reward
        sum_reward += total / len(seeds)
    print(f"success_rate={success:.3f}  avg_sum_reward={sum_reward:.2f}")
    if at_success:
        print(f"distance at success: mean {np.mean(at_success):.4f} m  "
              f"max {np.max(at_success):.4f} -> imputed_reward ~= {-np.mean(at_success):.3f}")
    return sum_reward, success


if __name__ == "__main__":
    from sim.flip.flip import FLIP
    from sim.flip.wristcam import WRISTCAM

    p = argparse.ArgumentParser()
    p.add_argument("version", choices=["flip", "wristcam"])
    version = {"flip": FLIP, "wristcam": WRISTCAM}[p.parse_args().version]
    eval_experts(version, range(900_000, 900_020))
