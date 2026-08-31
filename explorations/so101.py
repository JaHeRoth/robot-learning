# %%
import mujoco
from pathlib import Path
import os

# %%
path = os.path.expanduser("~/repos/mujoco_menagerie/trs_so_arm100/scene.xml")
model = mujoco.MjModel.from_xml_path(path)
data = mujoco.MjData(model)

# %%
print(f"{model.nq=}, {model.nu=}, {model.nbody=}")
print("\nJoint names:")
for i in range(model.nq):
    print("-", model.joint(i).name)
print("\nActuator names:")
for i in range(model.nu):
    print("-", model.actuator(i).name)
print("\nBody names:")
for i in range(model.nbody):
    print("-", model.body(i).name)

# %%
print(data.qpos)
mujoco.mj_resetDataKeyframe(model, data, 0)
print(data.qpos)

# %%
mujoco.mj_resetDataKeyframe(model, data, 0)
data.ctrl[0] = 1.0
print(f"{data.qpos=}")
print(f"{data.ctrl=}")
for i in range(1000):
    mujoco.mj_step(model, data)
    print(f"After {i + 1}/1000 steps (t={data.time}s): {data.qpos}")
print(f"{data.qpos=}")
print(f"{data.ctrl=}")

# %%
print(f"{data.body("Fixed_Jaw").xpos=}")
print(f"{data.body("Fixed_Jaw").xquat=}")
mujoco.mj_resetDataKeyframe(model, data, 0)
print(f"{data.body("Fixed_Jaw").xpos=}")
print(f"{data.body("Fixed_Jaw").xquat=}")
mujoco.mj_forward(model, data)
print(f"{data.body("Fixed_Jaw").xpos=}")
print(f"{data.body("Fixed_Jaw").xquat=}")

# %%
