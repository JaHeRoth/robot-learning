"""Generate scripted-expert demonstrations for the SO-100 flip task, wrist-camera version.

The glass starts upside down at a randomised spot; the arm grasps the stem from
the side, lifts, rolls it 180 degrees to upright, and sets it on the shelf.

The grasp is deliberately sideways rather than top-down. A top-down grasp would
end the flip with the gripper pointing up, holding the glass from underneath --
which puts the wrist below the shelf top and makes placement geometrically
impossible. Approaching horizontally means the flip is a pure wrist roll about
the approach axis, which moves neither the grasp point nor the approach
direction, and leaves the gripper still reaching in horizontally at the end.
"""
import argparse
import json
import math
import os
from multiprocessing import Pool, cpu_count
from pathlib import Path
from time import time

import mujoco
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENE = REPO_ROOT / "scenes/so100_flip_wristcam/scene.xml"
REPO_ID = "jaheroth/so100_flip_wristcam"
TASK = "Flip the glass onto the shelf"
# Observation key -> scene camera. The fixed camera sees the whole workspace but
# views the grasp from far away and part-occluded by the arm; the wrist camera
# sees the stem close up, which is where the failures are.
CAMERAS = {"observation.image": "front", "observation.image_wrist": "wrist"}

FPS = 25
IMG = (96, 96)
JAW, OPEN, CLOSE = 5, 1.2, -0.17
UP = np.array([0.0, 0.0, 1.0])

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
# Gripper stem axis at the grasp. This is the branch with the camera above the
# gripper; the flip inverts it, so placement uses the opposite.
GRASP_DIR = UP
LIFT_Z = 0.20
# The wrist roll may start once the grasp point is this high. The bowl hangs
# 0.083 m below it and the roll axis is offset another 0.036, so this is about as
# early as it can start without the bowl grazing the floor.
LIFT_CLEAR = 0.128
# The glass only has to be off the floor for the base to start turning; just the
# wrist roll needs the full LIFT_CLEAR margin.
LIFT_OFF = 0.11
SHELF_XY, SHELF_TOP = np.array([0.24, -0.16]), 0.10
# Spawn box, chosen so every point keeps >=5 cm of margin to the arm's
# reachability boundary at both the grasp and lift heights.
SPAWN_MID, SPAWN_HALF = np.array([0.0, -0.335]), np.array([0.10, 0.015])
# 0.80 is as fast as this arm goes with real torque headroom: at 0.65 the wrist
# pitch hits 3.2 of its 3.5 N.m limit. Both jitters only ever slow things down
# from there, so no demonstration is generated at the edge of saturation.
NOMINAL_SPEED = 0.80
SPEED_RANGE = (NOMINAL_SPEED, 1.05)
SEGMENT_JITTER = (1.0, 1.14)
VIA_WOBBLE = 0.02


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
    def __init__(self, model: mujoco.MjModel, img=IMG):
        self.m = model
        self.d = mujoco.MjData(model)
        self.scratch = mujoco.MjData(model)  # IK must never touch the live state
        self.renderer = mujoco.Renderer(model, height=img[0], width=img[1])
        self.key = [model.key(i).name for i in range(model.nkey)].index("start")
        self.lb, self.ub = model.jnt_range[:6, 0].copy(), model.jnt_range[:6, 1].copy()
        self.pad_f = model.geom("fixed_jaw_pad_1").id
        self.pad_m = model.geom("moving_jaw_pad_1").id
        self.tip = model.site("tip").id
        self.home = model.key("start").qpos[:6].copy()
        self.rest = model.key("rest").qpos[:6].copy()  # folded-up pose
        self.kp = float(model.actuator_gainprm[0, 0])
        self.n_substeps = round(1 / (FPS * model.opt.timestep))
        self.frames = []
        self.speed = NOMINAL_SPEED  # episode-wide multiplier on segment lengths
        self.jitter = None   # rng for per-segment timing and path-shape noise

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

    def clear_at(self, q):
        return not self.bad_contacts(self._load(q))

    def solve(self, pos, stem_dir, q0):
        """IK from several shoulder/elbow seeds, keeping the solution that reaches
        the target without putting the arm through the floor or the glass. The
        constraint set is square, so there is no nullspace to optimise inside --
        different postures are reachable only from different seeds. Scored against
        the seed rather than any fixed pose, so consecutive waypoints stay in the
        same branch and the arm does not snap between elbow configurations."""
        best = None
        for dp, de in [(0, 0), (-0.4, 0.4), (0.4, -0.4), (-0.7, 0.7), (0.3, 0.3), (-0.3, -0.3)]:
            s0 = q0.copy()
            s0[1], s0[2] = np.clip([s0[1] + dp, s0[2] + de], self.lb[1:3], self.ub[1:3])
            q, e, ang = self.ik(pos, stem_dir, s0)
            if e > 2e-3 or ang > 4.0:
                continue
            score = (bool(self.bad_contacts(self._load(q))),
                     float(np.linalg.norm(q[:5] - q0[:5])))
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

    def dur(self, n_frames):
        """Segment length in frames: the episode's speed factor, plus a little noise
        of its own. Fixed durations make every demonstration the same length, which
        is an easier regularity to latch onto than the task; a single episode-wide
        factor still leaves the segments in lockstep with each other."""
        n = n_frames * self.speed
        if self.jitter is not None:
            n *= self.jitter.uniform(*SEGMENT_JITTER)
        return max(2, round(n))

    def wobble(self, q, scale):
        """Nudge an intermediate waypoint. Applied only to via points, never to the
        grasp or the placement, so the path between them varies while the two poses
        that have to be exact stay exact."""
        if self.jitter is None:
            return q
        q = q.copy()
        q[:5] = np.clip(q[:5] + self.jitter.normal(0, scale, 5), self.lb[:5], self.ub[:5])
        return q

    def move_bezier(self, controls, jaw, n_frames, settle=0, record=True,
                    hold_roll_until_z=None, roll_delay=0.0, roll_until=0.88, jaw_to=None):
        """Bezier through the current pose and `controls` (the last is the target).

        With controls [above_pickup, above_shelf, shelf] the curve leaves the pickup
        heading straight up and arrives at the shelf heading straight down, because
        a Bezier's end tangents point at its neighbouring control points. Chaining
        straight segments instead produces visible corners where the motion switches
        from vertical to diagonal to vertical."""
        pts = [self.q_cmd.copy(), *controls]
        n = len(pts) - 1
        binom = [math.comb(n, i) for i in range(n + 1)]

        def bez(t):
            return sum(b * (1 - t) ** (n - k) * t ** k * p
                       for k, (b, p) in enumerate(zip(binom, pts)))

        # Only the wrist roll has to wait for height: it swings the bowl through an
        # arc below the grasp. Turning the base is free once the glass is off the
        # floor, so the two are scheduled separately.
        roll_from = 0.0
        if hold_roll_until_z is not None:
            for i in range(n_frames + 1):
                if self.grasp_point(self._load(bez(i / n_frames)))[2] >= hold_roll_until_z:
                    roll_from = i / n_frames
                    break
        roll_0, roll_1 = pts[0][4], pts[-1][4]

        for i in range(1, n_frames + settle + 1):
            if record:
                self.capture()
            t = min(i / n_frames, 1.0)
            self.q_cmd = bez(t)
            if hold_roll_until_z is not None:
                u = 0.0 if t <= roll_from else (t - roll_from) / max(1e-6, 1 - roll_from)
                self.q_cmd[4] = roll_0 + u * u * (3 - 2 * u) * (roll_1 - roll_0)
            elif roll_delay:
                # Turn inside a window rather than over the whole motion: late enough
                # that the jaws are off the ledge, done early enough that the camera
                # is back on top well before the arm folds in on itself.
                u = np.clip((t - roll_delay) / max(1e-6, roll_until - roll_delay), 0.0, 1.0)
                self.q_cmd[4] = roll_0 + u * u * (3 - 2 * u) * (roll_1 - roll_0)
            self.d.ctrl[:6] = self.servo(self.q_cmd)
            if jaw_to is None:
                self.d.ctrl[JAW] = jaw
            else:
                # Close on the way out, on the same delay as the wrist, so the arm
                # ends genuinely folded rather than retreating with its hand open.
                u = 0.0 if t <= roll_delay else (t - roll_delay) / max(1e-6, 1 - roll_delay)
                self.d.ctrl[JAW] = jaw + u * u * (3 - 2 * u) * (jaw_to - jaw)
            for _ in range(self.n_substeps):
                mujoco.mj_step(self.m, self.d)
            self.touches |= self.bad_contacts()

    def move_arc(self, q_via, q_target, jaw, n_frames, settle=0, record=True,
                 roll_delay=0.0):
        """Quadratic Bezier through q_via. Interpolating start -> via -> target as
        two straight legs makes the arm finish lifting before it begins turning;
        blending them means the lift, the base swing and the wrist roll all run at
        once, which reads as one motion rather than three."""
        q0 = self.q_cmd.copy()
        for i in range(1, n_frames + settle + 1):
            if record:
                self.capture()
            t = min(i / n_frames, 1.0)
            self.q_cmd = (1 - t) ** 2 * q0 + 2 * t * (1 - t) * q_via + t ** 2 * q_target
            if roll_delay:
                # Hold the wrist, then unwind it, so the jaws are already clear of
                # whatever they were next to before they start sweeping.
                u = 0.0 if t <= roll_delay else (t - roll_delay) / (1 - roll_delay)
                self.q_cmd[4] = q0[4] + u * u * (3 - 2 * u) * (q_target[4] - q0[4])
            self.d.ctrl[:6] = self.servo(self.q_cmd)
            self.d.ctrl[JAW] = jaw
            for _ in range(self.n_substeps):
                mujoco.mj_step(self.m, self.d)
            self.touches |= self.bad_contacts()

    def capture(self):
        frame = {
            "observation.state": self.d.qpos[:6].copy().astype(np.float32),
            "action": self.d.ctrl[:6].copy().astype(np.float32),
        }
        for key, cam in CAMERAS.items():
            self.renderer.update_scene(self.d, camera=cam)
            frame[key] = self.renderer.render().copy()
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

    def run(self, glass_xy, q_init, place_xy, strict=True):
        """Execute one demonstration. Returns True if the glass ends upright on the
        shelf. With strict=False the motion runs to the end regardless of failed
        checkpoints, which is what makes a diagnostic recording possible."""
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

        # Grasp the stem at its midpoint, with the stem hanging below the gripper.
        grasp_at = d.geom("glass_stem").xpos.copy() + GRASP_DZ * UP
        # Grasp with the wrist rolled so the camera rides above the gripper rather
        # than under it, where it would dig into the floor. Solving for that branch
        # directly does not converge from the home seed, and rolling a solved pose
        # afterwards drags the grip point off the stem -- the roll axis sits 3.6 cm
        # from the jaw. So solve the easy branch, roll that into a seed facing the
        # right way, and re-solve from there for a proper grip.
        q_easy, _, _, _ = self.solve(grasp_at, -UP, self.seed(glass_xy))
        seed_up = q_easy.copy()
        step = np.pi if q_easy[4] + np.pi <= self.ub[4] else -np.pi
        seed_up[4] = np.clip(q_easy[4] + step, self.lb[4], self.ub[4])
        q_grasp, e, ang, _ = self.solve(grasp_at, GRASP_DIR, seed_up)
        if self._gate(e <= 3e-3 and ang <= 4.0 and self.clear_at(q_grasp), "grasp_ik", strict):
            return False

        # Two waypoints, not one. The last leg must be purely horizontal so the jaw
        # slides in under the foot; descending diagonally onto the stem sweeps the
        # gripper's upper body straight through the foot's overhang.
        a_dir = self.approach_dir(q_grasp)
        for back in (ALIGN_BACK, 0.025, 0.02):
            q_align, e, ang, _ = self.solve(grasp_at - back * a_dir, GRASP_DIR, q_grasp)
            clear = self.clear_at(q_align)
            if e <= 3e-3 and ang <= 4.0 and clear:
                break
        if self._gate(e <= 3e-3 and ang <= 4.0 and clear, "align_ik", strict):
            return False

        pre_pos = grasp_at - BACKOFF * a_dir + PRE_LIFT * UP
        q_pre, e, ang, _ = self.solve(pre_pos, GRASP_DIR, q_align)
        if self._gate(e <= 3e-3 and ang <= 4.0 and self.clear_at(q_pre), "pregrasp_ik", strict):
            return False

        # Come in from above rather than swinging in at glass height: travelling
        # straight to the pre-grasp clips the bowl on the way. Descending behind the
        # glass gives maximum clearance past both the bowl and the foot.
        q_high, _, _, _ = self.solve(np.r_[pre_pos[:2], LIFT_Z], GRASP_DIR, q_pre)

        # Move in swung slightly off to the side so the open jaws hold a gap from
        # the stem instead of brushing it, then swing back to seat the grip point on
        # the stem once already in position.
        delta = self._swing_away(q_grasp, grasp_at, ENTRY_SWING) - q_grasp[BASE_ROT]

        q_high[JAW] = q_pre[JAW] = q_align[JAW] = q_grasp[JAW] = OPEN
        for q in (q_high, q_pre, q_align):
            q[BASE_ROT] = np.clip(q[BASE_ROT] + delta, self.lb[BASE_ROT], self.ub[BASE_ROT])
        self.move_to(q_high, OPEN, self.dur(16), settle=2)
        self.move_to(q_pre, OPEN, self.dur(14), settle=4)
        self.move_to(q_align, OPEN, self.dur(12), settle=6)
        self.move_to(q_grasp, OPEN, self.dur(14), settle=8)
        self.move_to(q_grasp, CLOSE, self.dur(12), settle=6)

        # The grasp is rigid from here, so the glass's pose in the tip frame is fixed.
        rel = self._tip_mat(d).T @ (self.glass().xpos - self.grasp_point())
        q_lift, e, _, _ = self.solve(np.r_[grasp_at[:2], LIFT_Z], GRASP_DIR, q_grasp)
        q_lift[JAW] = CLOSE

        # The flip is q4 += pi. Solving the shelf pose with the roll already applied
        # and arcing through the lift means the glass turns over while being carried.
        q_flip = q_lift.copy()
        q_flip[4] += np.pi if q_lift[4] + np.pi <= self.ub[4] else -np.pi
        q_above, e, clear = self.pose_for_glass(np.r_[place_xy, SHELF_TOP + 0.09], q_flip, rel, -GRASP_DIR)
        if self._gate(e <= 5e-3 and clear, "place_ik", strict):
            return False
        q_above[JAW] = CLOSE
        q_down, _, _ = self.pose_for_glass(np.r_[place_xy, SHELF_TOP + 0.001], q_above, rel, -GRASP_DIR)
        q_down[JAW] = CLOSE

        # Straight up first: the bowl hangs below the grasp, so it has to clear the
        # floor before any rotation starts. Then one curve all the way to the shelf.
        q_rise, _, _, _ = self.solve(np.r_[grasp_at[:2], LIFT_OFF], GRASP_DIR, q_grasp)
        q_rise[JAW] = CLOSE
        self.move_to(q_rise, CLOSE, self.dur(8), settle=1)
        self.move_bezier([self.wobble(q_lift, VIA_WOBBLE), q_above, q_down], CLOSE, self.dur(78), settle=10,
                         hold_roll_until_z=LIFT_CLEAR)
        if self._gate(self.tilt() >= 0.80, "flip", strict):
            return False
        self.move_to(q_down, OPEN, self.dur(10), settle=6)          # release

        # Only the moving jaw swings open; the fixed one stays against the stem, so
        # pulling straight back drags the glass off the shelf. Swinging the base
        # opens a sideways gap first, then withdraw through it.
        q_clear = q_down.copy()
        q_clear[BASE_ROT] = self._swing_away(q_down, self.glass().xpos, RELEASE_SWING)
        q_clear[JAW] = OPEN

        # Curve out through the gap rather than rotating in place and then retreating:
        # the arc still opens sideways first, which is what keeps the fixed jaw from
        # dragging the glass, but reads as one motion.
        s_clear = self._load(q_clear)
        gp = self.grasp_point(s_clear)
        # Retreat back toward the base. Anything else -- along the gripper axis, or
        # radially away from the glass -- can carry the jaw up through the bowl,
        # since the placement pose's compass direction is not pinned.
        toward_base = np.array([-gp[0], -gp[1], 0.0])
        toward_base /= max(1e-9, np.linalg.norm(toward_base))
        back_pos = gp + 0.10 * toward_base + 0.02 * UP
        q_back, _, _, _ = self.solve(back_pos, -GRASP_DIR, q_clear)
        # Unwind the half turn on the way out: left inverted, the camera swings into
        # the base as the arm folds up. The jaw is empty by now, so the roll is free.
        turn = np.pi / 2 if q_back[4] + np.pi / 2 <= self.ub[4] else -np.pi / 2
        q_back[4] = np.clip(q_back[4] + turn, self.lb[4], self.ub[4])
        q_back[JAW] = OPEN
        # Curve out through the retreat and on into a folded pose, so the elbow
        # starts closing while the gripper is still on its way back rather than
        # after it has stopped.
        # Retreat, folding partway on the way out, then finish into the model's rest
        # pose. Curving straight from the shelf to rest instead sweeps the arm back
        # across the shelf, since rest faces the other way. The wrist turns as soon
        # as the jaws are off the ledge and can no longer catch it.
        # Open the elbow while backing off the ledge: pulling straight back with it
        # already folding keeps the jaws low and close to the shelf edge. Extending
        # first lifts them clear, then the arm folds up.
        q_open = q_back.copy()
        q_open[2] = np.clip(q_back[2] - 0.35, self.lb[2], self.ub[2])
        q_mid = q_back.copy()
        q_mid[1:4] = q_back[1:4] + 0.45 * (self.rest[1:4] - q_back[1:4])
        self.move_bezier([q_clear, self.wobble(q_open, VIA_WOBBLE), q_back, q_mid], OPEN, self.dur(50), settle=4,
                         roll_delay=0.23, roll_until=0.88, jaw_to=CLOSE)
        self.move_to(self.rest, CLOSE, self.dur(26), settle=6)
        for _ in range(50):
            mujoco.mj_step(m, d)
        self.capture()

        g = self.glass().xpos
        placed = (self.tilt() > 0.9 and abs(g[2] - SHELF_TOP) < 0.03
                  and np.linalg.norm(g[:2] - SHELF_XY) < 0.06)
        self._gate(placed, "placed", strict)
        # Nothing but the pads may have touched the glass, and nothing else the scene.
        self._gate(not self.touches, "touched", strict)
        return bool(placed and not self.touches)


def sample_episode(rng, expert, randomize_speed=False):
    """Glass pose, arm start and place spot. The spawn box is pre-vetted for
    reachability margin, so no rejection sampling is needed."""
    expert.speed = rng.uniform(*SPEED_RANGE) if randomize_speed else NOMINAL_SPEED
    expert.jitter = np.random.default_rng(rng.integers(2**32)) if randomize_speed else None
    glass_xy = SPAWN_MID + rng.uniform(-SPAWN_HALF, SPAWN_HALF)
    q_init = np.clip(expert.home + rng.normal(0, 0.05, 6), expert.lb, expert.ub)
    q_init[JAW] = OPEN
    place_xy = SHELF_XY  # always the same spot: vary the motion, not the target
    return glass_xy, q_init, place_xy


_worker: dict = {}


def _init_worker():
    _worker["expert"] = Expert(mujoco.MjModel.from_xml_path(str(SCENE)))


def _episode(args):
    """One demonstration, or None if it did not end with the glass upright on the
    shelf and nothing but the pads touched. Judged on that outcome rather than on
    the intermediate IK checks: those are conservative and abort runs that go on to
    succeed, which would drop exactly the near-edge spawns the policy needs to see."""
    seed, randomize_speed = args
    expert = _worker["expert"]
    glass_xy, q_init, place_xy = sample_episode(
        np.random.default_rng(seed), expert, randomize_speed)
    if not expert.run(glass_xy, q_init, place_xy, strict=False):
        return None
    return expert.frames


def generate(n_episodes=1000, seed=0, workers=None, out_dir=None, randomize_speed=False):
    """Write n_episodes demonstrations to out_dir, resuming if it already holds some.

    Seeds are consumed in order and the next unused one is checkpointed after every
    episode, so an interrupted run picks up exactly where it stopped instead of
    regenerating from scratch. Pass disjoint `seed` ranges to shard across machines
    or across several processes on one machine.
    """
    root = Path(out_dir) if out_dir else REPO_ROOT / f"outputs/so100_flip_wristcam_{int(time())}"
    progress = root / "progress.json"
    if progress.exists():
        state = json.loads(progress.read_text())
        dataset = LeRobotDataset(REPO_ID, root=root)
        n_ok, next_seed = dataset.num_episodes, state["next_seed"]
        print(f"resuming {root}: {n_ok} episodes done, next seed {next_seed}")
    else:
        model = mujoco.MjModel.from_xml_path(str(SCENE))
        joint_names = [model.joint(i).name for i in range(6)]
        dataset = LeRobotDataset.create(
            repo_id=REPO_ID,
            fps=FPS,
            features={
                **{k: {"dtype": "video", "shape": (*IMG, 3),
                       "names": ["height", "width", "channels"]} for k in CAMERAS},
                "observation.state": {"dtype": "float32", "shape": (6,), "names": joint_names},
                "action": {"dtype": "float32", "shape": (6,), "names": joint_names},
            },
            root=root,
        )
        n_ok, next_seed = 0, seed

    # Oversample: some fraction of draws fail the expert's checks and are dropped.
    workers = workers or max(1, min(cpu_count(), 32))
    remaining = n_episodes - n_ok
    if remaining <= 0:
        return dataset, n_ok
    seeds = range(next_seed, next_seed + int(remaining * 1.4) + 16)
    n_tried = 0
    with Pool(workers, initializer=_init_worker) as pool:
        with tqdm(total=n_episodes, initial=n_ok) as bar:
            # ordered imap, so "every seed below next_seed is done" stays true
            jobs = ((s, randomize_speed) for s in seeds)
            for offset, frames in enumerate(pool.imap(_episode, jobs, chunksize=1)):
                n_tried += 1
                if frames is not None:
                    for frame in frames:
                        dataset.add_frame(frame, task=TASK)
                    dataset.save_episode()
                    n_ok += 1
                    bar.update(1)
                progress.write_text(json.dumps({"next_seed": seeds[offset] + 1, "n_ok": n_ok}))
                if n_ok >= n_episodes:
                    break
    made = n_ok - (n_episodes - remaining)
    print(f"{n_ok} episodes total; this run made {made} from {n_tried} attempts "
          f"({100*made/max(n_tried,1):.0f}% clean)")
    return dataset, n_ok


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--out-dir", default=None, help="resumed in place if it already exists")
    p.add_argument("--seed", type=int, default=0, help="first seed; use disjoint ranges to shard")
    p.add_argument("--randomize-speed", action="store_true",
                   help="vary how fast each demonstration executes")
    args = p.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    ds, n = generate(n_episodes=args.episodes, seed=args.seed, workers=args.workers,
                     out_dir=args.out_dir, randomize_speed=args.randomize_speed)
    print(f"done: {n} episodes at {ds.root}")
