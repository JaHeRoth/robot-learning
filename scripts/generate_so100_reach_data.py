"""Generate scripted-expert demonstrations for the SO-100 reach task.

Samples reachable goal poses (with rejection filtering), places the goal cube
at the target fingertip position, records interpolated reach trajectories at
25 Hz into a LeRobotDataset.
"""
from pathlib import Path
from time import time

import mujoco
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent


def sample_q_init_and_target(
    rng: np.random.Generator,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
) -> tuple[np.ndarray, np.ndarray]:
    lb, ub = model.jnt_range[:, 0], model.jnt_range[:, 1]
    center, span = (lb + ub) / 2, ub - lb
    home = model.key("home").qpos

    q_init = (
        home
        + rng.normal(
            loc=0.0,
            scale=np.clip(np.minimum(home - lb, ub - home) / 2, max=0.05),
            size=model.njnt,
        )
    ).clip(lb, ub)

    for _ in range(100):
        q_target = rng.uniform(
            low=center - 0.35 * span, high=center + 0.35 * span
        )
        data.qpos = q_target.copy()
        mujoco.mj_forward(model, data)
        data.mocap_pos[0] = data.site("tip").xpos.copy()

        data.qpos = q_init.copy()
        mujoco.mj_forward(model, data)

        underground = data.mocap_pos[0][2] < 0.05
        cramped = np.hypot(data.mocap_pos[0][0], data.mocap_pos[0][1]) < 0.12

        renderer.update_scene(data, camera="front")
        img = renderer.render()
        cube_mask = (
            (img[:, :, 0] > 150)
            & (img[:, :, 1] < 100)
            & (img[:, :, 2] < 100)
        )
        cube_visible = (
            cube_mask[5:-5, 5:-5].sum() == cube_mask.sum()
            and cube_mask.sum() >= 30
        )
        if not underground and not cramped and cube_visible:
            return q_init, q_target
    raise RuntimeError("goal sampling failed 100x")


def capture_and_step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    dataset: LeRobotDataset,
    task: str,
) -> None:
    renderer.update_scene(data, camera="front")
    dataset.add_frame(
        {
            "observation.image": renderer.render(),
            "observation.state": data.qpos.copy().astype(np.float32),
            "action": data.ctrl.copy().astype(np.float32),
        },
        task=task,
    )
    for _ in range(20):
        mujoco.mj_step(model, data)


def generate_expert_trajectory(
    rng: np.random.Generator,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    dataset: LeRobotDataset,
    task: str,
) -> None:
    mujoco.mj_resetData(model, data)
    ctrl_steps = rng.integers(low=50, high=100)
    q_init, q_target = sample_q_init_and_target(rng, model, data, renderer)

    data.qpos = q_init.copy()
    mujoco.mj_forward(model, data)

    for step in range(1, ctrl_steps + 1):
        data.ctrl = q_init + (step / ctrl_steps) * (q_target - q_init)
        capture_and_step(model, data, renderer, dataset, task)
    for _ in range(10):
        capture_and_step(model, data, renderer, dataset, task)
    dataset.save_episode()


def generate_expert_data() -> LeRobotDataset:
    num_episodes = 100
    model = mujoco.MjModel.from_xml_path(
        str(REPO_ROOT / "scenes/so100_reach/scene.xml")
    )
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=96, width=96)
    rng = np.random.default_rng(seed=0)

    joint_names = [model.joint(i).name for i in range(model.njnt)]
    dataset = LeRobotDataset.create(
        repo_id="jaheroth/so100_reach",
        fps=25,
        features={
            "observation.image": {
                "dtype": "video",
                "shape": (96, 96, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (6,),
                "names": joint_names,
            },
            "action": {
                "dtype": "float32",
                "shape": (6,),
                "names": joint_names,
            },
        },
        root=REPO_ROOT / f"outputs/so100_reach_{int(time())}",
    )

    for _ in tqdm(range(num_episodes)):
        generate_expert_trajectory(
            rng, model, data, renderer, dataset, task="Reach the red cube"
        )
    return dataset


if __name__ == "__main__":
    dataset = generate_expert_data()
    print(f"done: {dataset.num_episodes} episodes at {dataset.root}")
