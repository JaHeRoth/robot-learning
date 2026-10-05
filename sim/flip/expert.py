"""Scripted expert for the SO-100 flip task, shared by both versions.

The glass starts upside down at a randomised spot; the arm grasps the stem from
the side, lifts, rolls it 180 degrees to upright, and sets it on the shelf.

The grasp is deliberately sideways rather than top-down. A top-down grasp would
end the flip with the gripper pointing up, holding the glass from underneath --
which puts the wrist below the shelf top and makes placement geometrically
impossible. Approaching horizontally means the flip is a pure wrist roll about
the approach axis, which moves neither the grasp point nor the approach
direction, and leaves the gripper still reaching in horizontally at the end.

Each version (flip.py, wristcam.py) subclasses Expert with its own motion in `run`.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np

TASK = "Flip the glass onto the shelf"

FPS = 25
IMG = (96, 96)
JAW, OPEN, CLOSE = 5, 1.2, -0.17
UP = np.array([0.0, 0.0, 1.0])
FRONT = {"observation.image": "front"}

# Gripper frame. The jaws close along JAW_AXIS, and a grasped vertical stem lies
# along STEM_AXIS; rolling about APPROACH is what turns the glass over.
APPROACH = np.array([0.0, -1.0, 0.0])
JAW_AXIS = np.array([-1.0, 0.0, 0.0])
STEM_AXIS = np.cross(APPROACH, JAW_AXIS)

# The pre-grasp sits back along the approach axis and a little above it. Backing
# off in pure horizontal would land inside the arm's near-field dead zone, where
# the gripper cannot be placed at stem height at all.
BACKOFF, PRE_LIFT, ALIGN_BACK = 0.05, 0.04, 0.03
# Swinging the base is what opens a sideways gap between the jaws and the stem.
# The arm has no wrist DoF in that direction: roll only spins the hand about its
# own axis, and pitch tilts it in the vertical plane.
BASE_ROT = 0
ENTRY_SWING, RELEASE_SWING = 0.09, 0.09
GRASP_DZ = -0.005  # below the stem midpoint, buying clearance from the foot
LIFT_Z = 0.20
SHELF_XY, SHELF_TOP = np.array([0.24, -0.16]), 0.10
# Spawn box, chosen so every point keeps >=5 cm of margin to the arm's
# reachability boundary at both the grasp and lift heights.
SPAWN_MID, SPAWN_HALF = np.array([0.0, -0.335]), np.array([0.10, 0.015])


@dataclass(frozen=True)
class Version:
    """Everything the generator and the eval env need to know about one version."""
    name: str                 # prefix of the default output folder
    scene: Path
    repo_id: str
    cameras: dict[str, str]   # observation key -> scene camera
    expert: type              # Expert subclass
    sample_episode: Callable  # (rng, expert, **kwargs) -> (glass_xy, q_init, place_xy)
    horizon: int              # eval episode length


def servo_ctrl(model, scratch, live_qpos, q, kp):
    """Control signal that actually holds pose q. The joints are P servos
    (force = kp*(ctrl - qpos) - kv*qvel), so gravity leaves a steady-state droop --
    12 degrees at the wrist, enough to miss the stem by 12 cm. Offsetting the
    command by the gravity torque cancels it. Shared with the eval env so a rollout
    starts from the same settled pose the demonstrations do."""
    scratch.qpos[:] = live_qpos
    scratch.qpos[:6] = q
    scratch.qvel[:] = 0
    mujoco.mj_forward(model, scratch)
    return q + scratch.qfrc_bias[:6] / kp


class Expert:
    def __init__(self, model: mujoco.MjModel, cameras=FRONT, img=IMG):
        self.m = model
        self.d = mujoco.MjData(model)
        self.scratch = mujoco.MjData(model)  # IK must never touch the live state
        self.renderer = mujoco.Renderer(model, height=img[0], width=img[1])
        self.cameras = cameras
        self.key = [model.key(i).name for i in range(model.nkey)].index("start")
        self.lb, self.ub = model.jnt_range[:6, 0].copy(), model.jnt_range[:6, 1].copy()
        self.pad_f = model.geom("fixed_jaw_pad_1").id
        self.pad_m = model.geom("moving_jaw_pad_1").id
        self.tip = model.site("tip").id
        self.home = model.key("start").qpos[:6].copy()
        self.kp = float(model.actuator_gainprm[0, 0])
        self.n_substeps = round(1 / (FPS * model.opt.timestep))
        self.frames = []

        self.glass_geoms = {model.geom(i).id for i in range(model.ngeom)
                            if model.geom(i).name.startswith("glass")}
        self.world_geoms = {model.geom("floor").id, model.geom("shelf").id}
        self.pads = {model.geom(i).id for i in range(model.ngeom)
                     if "jaw_pad" in model.geom(i).name}
        self.grip = self._grip_angle()

    def _grip_angle(self):
        """Jaw angle at which the pads just span the stem. IK has to aim the pad
        midpoint evaluated here, not with the jaw open: the fixed jaw barely moves
        while the moving one sweeps ~10 cm, so the open midpoint sits 5 cm from
        where the stem actually ends up being held."""
        want = 2 * (float(self.m.geom("fixed_jaw_pad_1").size[0])
                    + float(self.m.geom("glass_stem").size[0]))
        q = self.home.copy()
        gaps = []
        for jaw in np.linspace(CLOSE, OPEN, 200):
            q[JAW] = jaw
            s = self._load(q)
            gaps.append(np.linalg.norm(s.geom_xpos[self.pad_f] - s.geom_xpos[self.pad_m]))
        return float(np.linspace(CLOSE, OPEN, 200)[np.argmin(np.abs(np.array(gaps) - want))])

    def bad_contacts(self, data=None):
        """Contacts other than the pads on the stem, or the glass resting on the
        world. Anything else means the arm is bumping into the scene."""
        data = data if data is not None else self.d
        out = set()
        for c in data.contact[:data.ncon]:
            g1, g2 = int(c.geom1), int(c.geom2)
            pair = {g1, g2}
            if pair <= self.glass_geoms | self.world_geoms:
                continue                                  # glass on floor/shelf
            if pair <= self.pads | self.glass_geoms:
                continue                                  # gripper on the glass
            out.add(tuple(sorted((self.m.geom(g1).name, self.m.geom(g2).name))))
        return out

    def grasp_point(self, data=None):
        data = data if data is not None else self.d
        return (data.geom_xpos[self.pad_f] + data.geom_xpos[self.pad_m]) / 2

    def _tip_mat(self, data):
        return data.site_xmat[self.tip].reshape(3, 3)

    def _load(self, q):
        s = self.scratch
        s.qpos[:] = self.d.qpos
        s.qpos[:6] = q
        mujoco.mj_forward(self.m, s)
        return s

    def approach_dir(self, q):
        return self._tip_mat(self._load(q)) @ APPROACH

    def seed(self, xy):
        q = self.home.copy()
        q[0] = np.clip(np.arctan2(xy[0], -xy[1]), self.lb[0], self.ub[0])
        q[JAW] = OPEN
        return q

    def posture_ref(self, q0):
        """Pose that solve() prefers solutions close to."""
        return self.home

    def solve(self, pos, stem_dir, q0):
        """IK from several shoulder/elbow seeds, keeping the solution that reaches
        the target without putting the arm through the floor or the glass. The
        constraint set is square, so there is no nullspace to optimise inside --
        different postures are reachable only from different seeds."""
        ref = self.posture_ref(q0)
        best = None
        for dp, de in [(0, 0), (-0.4, 0.4), (0.4, -0.4), (-0.7, 0.7), (0.3, 0.3), (-0.3, -0.3)]:
            s0 = q0.copy()
            s0[1], s0[2] = np.clip([s0[1] + dp, s0[2] + de], self.lb[1:3], self.ub[1:3])
            q, e, ang = self.ik(pos, stem_dir, s0)
            if e > 2e-3 or ang > 4.0:
                continue
            score = (bool(self.bad_contacts(self._load(q))),
                     float(np.linalg.norm(q[:5] - ref[:5])))
            if best is None or score < best[0]:
                best = (score, q, e, ang)
        if best is None:
            return self.ik(pos, stem_dir, q0) + (False,)
        return best[1], best[2], best[3], not best[0][0]

    def pose_for_glass(self, target, q_seed, rel, stem_dir):
        """Arm pose that puts the *glass* at target, upright. The grasp is rigid, so
        the glass sits at grasp_point + R_tip @ rel; R_tip depends on the pose being
        solved, so iterate. Lets the shelf poses be computed before the flip has
        physically happened, which is what allows rolling and carrying at once."""
        q, e, clear = q_seed, np.inf, False
        for _ in range(3):
            R = self._tip_mat(self._load(q))
            q, e, _, clear = self.solve(target - R @ rel, stem_dir, q)
        return q, e, clear

    def ik(self, pos, stem_dir, q0, iters=2000, w=0.6):
        """Damped least squares on the grasp point, with the grasped stem aligned to
        stem_dir. Pinning the stem (2 DoF) plus position (3) uses the arm's 5 DoF
        exactly, and implies a horizontal approach without pinning its compass
        direction -- the leftover spin is about the glass's own axis, so it costs
        nothing. Fixing the roll after solving instead does not work: the roll axis
        is offset ~3.6 cm from the jaw midpoint, so rolling drags the grasp point."""
        s = self.scratch
        s.qpos[:] = self.d.qpos
        s.qpos[:6] = q0
        s.qpos[JAW] = self.grip
        Jf, Jm, Jr = (np.zeros((3, self.m.nv)) for _ in range(3))
        for _ in range(iters):
            mujoco.mj_forward(self.m, s)
            mujoco.mj_jacGeom(self.m, s, Jf, None, self.pad_f)
            mujoco.mj_jacGeom(self.m, s, Jm, None, self.pad_m)
            mujoco.mj_jacSite(self.m, s, None, Jr, self.tip)
            n = self._tip_mat(s) @ STEM_AXIS
            err = np.r_[pos - self.grasp_point(s), w * np.cross(n, stem_dir)]
            J = np.vstack([((Jf + Jm) / 2)[:, :6], w * Jr[:, :6]])
            J[:, JAW] = 0
            if np.linalg.norm(err) < 2e-4:
                break
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(len(err)), err)
            s.qpos[:6] = np.clip(s.qpos[:6] + 0.4 * dq, self.lb, self.ub)
        mujoco.mj_forward(self.m, s)
        n = self._tip_mat(s) @ STEM_AXIS
        return (s.qpos[:6].copy(),
                float(np.linalg.norm(pos - self.grasp_point(s))),
                float(np.degrees(np.arccos(np.clip(n @ stem_dir, -1, 1)))))

    def servo(self, q):
        return servo_ctrl(self.m, self.scratch, self.d.qpos, q, self.kp)

    def move_to(self, q_target, jaw, n_frames, settle=0, record=True):
        """Ramp to a pose over n_frames control frames, then hold for `settle` more.
        The hold matters: the servos lag the ramp, and arriving 14 mm high is the
        difference between grasping the stem and knocking the glass's foot."""
        q_start = self.q_cmd.copy()
        for i in range(1, n_frames + settle + 1):
            if record:
                self.capture()
            self.q_cmd = q_start + min(i / n_frames, 1.0) * (q_target - q_start)
            self.d.ctrl[:6] = self.servo(self.q_cmd)
            self.d.ctrl[JAW] = jaw
            for _ in range(self.n_substeps):
                mujoco.mj_step(self.m, self.d)
            self.touches |= self.bad_contacts()

    def _swing_away(self, q, from_pos, angle):
        """Base rotation, swung whichever way clears the jaws of from_pos. Score the
        nearest pad, not the jaw midpoint: with the jaw open the midpoint sits 5 cm
        to one side, so scoring it picks the direction that drives a finger into the
        stem while the midpoint appears to move away."""
        lo, hi = self.lb[BASE_ROT], self.ub[BASE_ROT]
        cands = [np.clip(q[BASE_ROT] + s * angle, lo, hi) for s in (1, -1)]

        def clearance(r):
            s = self._load(np.r_[r, q[BASE_ROT + 1:5], OPEN])
            return min(np.linalg.norm(s.geom_xpos[g] - from_pos) for g in self.pads)
        return max(cands, key=clearance)

    def capture(self):
        frame = {
            "observation.state": self.d.qpos[:6].copy().astype(np.float32),
            "action": self.d.ctrl[:6].copy().astype(np.float32),
        }
        for key, cam in self.cameras.items():
            self.renderer.update_scene(self.d, camera=cam)
            frame[key] = self.renderer.render()
        self.frames.append(frame)

    def glass(self):
        return self.d.body("glass")

    def tilt(self):
        return float(self.glass().xmat.reshape(3, 3)[2, 2])

    def _gate(self, ok, name, strict):
        """Record a failed checkpoint; abort only when running strict."""
        if not ok:
            self.failures.append(name)
        return not ok and strict

    def _start(self, glass_xy, q_init):
        """Reset to the episode's start pose and let the glass settle."""
        m, d = self.m, self.d
        self.failures, self.touches = [], set()
        mujoco.mj_resetDataKeyframe(m, d, self.key)
        d.qpos[:6] = q_init
        d.qpos[6:8] = glass_xy
        mujoco.mj_forward(m, d)
        self.q_cmd = q_init.copy()
        d.ctrl[:6] = self.servo(q_init)
        for _ in range(100):
            mujoco.mj_step(m, d)  # let the glass settle
        self.frames = []

    def _finish(self, strict):
        """Let everything come to rest, record the last frame and judge the outcome."""
        for _ in range(50):
            mujoco.mj_step(self.m, self.d)
        self.capture()

        g = self.glass().xpos
        placed = (self.tilt() > 0.9 and abs(g[2] - SHELF_TOP) < 0.03
                  and np.linalg.norm(g[:2] - SHELF_XY) < 0.06)
        self._gate(placed, "placed", strict)
        # Nothing but the pads may have touched the glass, and nothing else the scene.
        self._gate(not self.touches, "touched", strict)
        return bool(placed and not self.touches)

    def run(self, glass_xy, q_init, place_xy, strict=True):
        """Execute one demonstration. Returns True if the glass ends upright on the
        shelf. With strict=False the motion runs to the end regardless of failed
        checkpoints, which is what makes a diagnostic recording possible."""
        raise NotImplementedError
