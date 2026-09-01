# %%
import mujoco
from pathlib import Path
import os
import numpy as np
from matplotlib import pyplot as plt

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
# print(f"{data.body("Fixed_Jaw").xquat=}")
mujoco.mj_resetDataKeyframe(model, data, 0)
print(f"{data.body("Fixed_Jaw").xpos=}")
# print(f"{data.body("Fixed_Jaw").xquat=}")
mujoco.mj_forward(model, data)
print(f"{data.body("Fixed_Jaw").xpos=}")
# print(f"{data.body("Fixed_Jaw").xquat=}")
data.qpos[0] = 1.0
print(f"{data.body("Fixed_Jaw").xpos=}")
mujoco.mj_forward(model, data)
print(f"{data.body("Fixed_Jaw").xpos=}")
data.qpos[1] = 0.0
mujoco.mj_forward(model, data)
print(f"{data.body("Fixed_Jaw").xpos=}")
data.qpos[1] = 0.174
mujoco.mj_forward(model, data)
print(f"{data.body("Fixed_Jaw").xpos=}")
data.qpos = np.zeros(6)
mujoco.mj_forward(model, data)
print(f"{data.body("Fixed_Jaw").xpos=}")
data.qpos[0] = 1.57
mujoco.mj_forward(model, data)
print(f"{data.body("Fixed_Jaw").xpos=}")
mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)
print(f"{data.body("Fixed_Jaw").xpos=}")
data.qpos[0] = 1.70
mujoco.mj_forward(model, data)
print(f"{data.body("Fixed_Jaw").xpos=}")
# Observations:
# - Coordinates are in meters
# - Right hand rule applies here for determining
#  xyz and their positive directions.
# - Rotation joint's axis isn't at origin

# %%
def plot_angle_against_target(period: int):
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    pitch_range = model.joint("Pitch").range
    pitch_ctrls = (
        pitch_range.mean()
        + 0.6 * (pitch_range[1] - pitch_range[0]) / 2
        * np.sin(2 * np.pi * np.arange(10000) / period)
    )
    qposes = []
    for pitch_ctrl in pitch_ctrls:
        data.ctrl[1] = pitch_ctrl
        mujoco.mj_step(model, data)
        qposes.append(data.qpos.copy())
    pitch_qposes = np.array([qp[1] for qp in qposes])

    plt.plot(pitch_ctrls, label="Target")
    plt.plot(pitch_qposes, label="Angle")
    plt.xlabel("Time step")
    plt.ylabel("Pitch")
    plt.title(f"{period=}")
    plt.grid()
    plt.legend()
    plt.show()

    plt.plot(pitch_ctrls, pitch_qposes)
    plt.xlabel("Pitch target")
    plt.ylabel("Pitch angle")
    plt.title(f"{period=}")
    plt.grid()
    plt.show()

# Force clipping causes overshooting when oscillation is too fast
plot_angle_against_target(2000)
plot_angle_against_target(1000)
plot_angle_against_target(500)
plot_angle_against_target(250)
model.actuator_forcerange *= 100
# Can follow when there's no force clipping
plot_angle_against_target(500)
# But will then behave like low-pass filter when oscillation is too fast, due to gradual chasing
plot_angle_against_target(64)
model.actuator_forcerange /= 100
plot_angle_against_target(64)

# %%
