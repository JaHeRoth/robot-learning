# %%
import mujoco
import numpy as np
from matplotlib import pyplot as plt

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
