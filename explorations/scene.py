# %%
import mujoco
import numpy as np
from matplotlib import pyplot as plt
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from time import time

# %%
path = "scenes/so100_reach/scene.xml"
model = mujoco.MjModel.from_xml_path(path)
data = mujoco.MjData(model)

# %%
renderer = mujoco.Renderer(model, height=96, width=96)

# %%
def capture():
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="front")
    frame = renderer.render()
    plt.imshow(frame)
    print(f"{data.body("target").xpos=}")

data.mocap_pos[0] = np.array([0.2, 0, 0.02])
capture()

# %%
data.mocap_pos[0] = np.array([0.3, 0.1, 0.19])
capture()

# %%
def capture_ax(ax, title=""):
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="front")
    ax.imshow(renderer.render())
    ax.set_title(title, fontsize=8)
    ax.axis("off")

poses = {
    "home": model.key("home").qpos,
    "rest": model.key("rest").qpos,
    "zero": np.zeros(model.nq),
}
cube_positions = [
    (0.25, -0.25, 0.06), (-0.25, -0.2, 0.12), (0.0, -0.38, 0.22)
]

fig, axes = plt.subplots(3, 3, figsize=(6, 6))
for i, (pose_name, q) in enumerate(poses.items()):
    for j, cube_pos in enumerate(cube_positions):
        data.qpos[:] = q
        data.qvel[:] = 0
        data.mocap_pos[0] = cube_pos
        capture_ax(axes[i][j], f"{pose_name}, cube {j}")
fig.tight_layout()
plt.show()

# %%
mujoco.mj_resetData(model, data)
joint_names = [model.joint(i).name for i in range(model.njnt)]
task = "Reach the red cube"
dataset = LeRobotDataset.create(
    "jaheroth/so100_reach",
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
    root=f"outputs/so100_reach_{int(time())}"
)
rng = np.random.default_rng(0)
lb, ub = model.jnt_range[:, 0], model.jnt_range[:, 1]
center, span = (lb + ub) / 2, ub - lb

q_init = (
    model.key("home").qpos
    + rng.normal(
        loc=0.0,
        scale=np.clip(
            np.minimum(
                model.key("home").qpos - lb,
                ub - model.key("home").qpos,
            ) / 2,
            max=0.05,
        ),
        size=len(joint_names),
    )
).clip(lb, ub)

for _ in range(100):
    q_target = rng.uniform(
        center - 0.35 * span, center + 0.35 * span
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
        break
if underground or cramped or not cube_visible:
    raise Exception

ctrl_steps = rng.integers(50, 100)
for step in range(1, ctrl_steps + 1):
    data.ctrl = q_init + (step / ctrl_steps) * (q_target - q_init)
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
for _ in range(10):
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
dataset.save_episode()


# %%
