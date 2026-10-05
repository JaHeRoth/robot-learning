"""Generate scripted-expert demonstrations for the SO-100 flip task.

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
import os
from multiprocessing import Pool, cpu_count
from pathlib import Path
from time import time

import mujoco
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE = REPO_ROOT / "sim/flip/scene.xml"
REPO_ID = "jaheroth/so100_flip"
TASK = "Flip the glass onto the shelf"

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
LIFT_Z = 0.20
# Rise this far straight up before any rotation starts. The bowl hangs 0.083 m
# below the grasp point and the roll axis is offset another 0.036, so starting to
# turn any lower swings the bowl into the floor.
LIFT_CLEAR = 0.15
SHELF_XY, SHELF_TOP = np.array([0.24, -0.16]), 0.10
# Spawn box, chosen so every point keeps >=5 cm of margin to the arm's
# reachability boundary at both the grasp and lift heights.
SPAWN_MID, SPAWN_HALF = np.array([0.0, -0.335]), np.array([0.10, 0.015])


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

    def solve(self, pos, stem_dir, q0):
        """IK from several shoulder/elbow seeds, keeping the solution that reaches
        the target without putting the arm through the floor or the glass. The
        constraint set is square, so there is no nullspace to optimise inside --
        different postures are reachable only from different seeds."""
        best = None
        for dp, de in [(0, 0), (-0.4, 0.4), (0.4, -0.4), (-0.7, 0.7), (0.3, 0.3), (-0.3, -0.3)]:
            s0 = q0.copy()
            s0[1], s0[2] = np.clip([s0[1] + dp, s0[2] + de], self.lb[1:3], self.ub[1:3])
            q, e, ang = self.ik(pos, stem_dir, s0)
            if e > 2e-3 or ang > 4.0:
                continue
            score = (bool(self.bad_contacts(self._load(q))),
                     float(np.linalg.norm(q[:5] - self.home[:5])))
            if best is None or score < best[0]:
                best = (score, q, e, ang)
        if best is None:
            return self.ik(pos, stem_dir, q0) + (False,)
        return best[1], best[2], best[3], not best[0][0]

    def pose_for_glass(self, target, q_seed, rel):
        """Arm pose that puts the *glass* at target, upright. The grasp is rigid, so
        the glass sits at grasp_point + R_tip @ rel; R_tip depends on the pose being
        solved, so iterate. Lets the shelf poses be computed before the flip has
        physically happened, which is what allows rolling and carrying at once."""
        q, e, clear = q_seed, np.inf, False
        for _ in range(3):
            R = self._tip_mat(self._load(q))
            q, e, _, clear = self.solve(target - R @ rel, UP, q)
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

    def move_arc(self, q_via, q_target, jaw, n_frames, settle=0, record=True):
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
            self.d.ctrl[:6] = self.servo(self.q_cmd)
            self.d.ctrl[JAW] = jaw
            for _ in range(self.n_substeps):
                mujoco.mj_step(self.m, self.d)
            self.touches |= self.bad_contacts()

    def capture(self):
        self.renderer.update_scene(self.d, camera="front")
        self.frames.append({
            "observation.image": self.renderer.render(),
            "observation.state": self.d.qpos[:6].copy().astype(np.float32),
            "action": self.d.ctrl[:6].copy().astype(np.float32),
        })

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
        q_grasp, e, ang, clear = self.solve(grasp_at, -UP, self.seed(glass_xy))
        if self._gate(e <= 3e-3 and ang <= 4.0 and clear, "grasp_ik", strict):
            return False

        # Two waypoints, not one. The last leg must be purely horizontal so the jaw
        # slides in under the foot; descending diagonally onto the stem sweeps the
        # gripper's upper body straight through the foot's overhang.
        a_dir = self.approach_dir(q_grasp)
        for back in (ALIGN_BACK, 0.025, 0.02):
            q_align, e, ang, clear = self.solve(grasp_at - back * a_dir, -UP, q_grasp)
            if e <= 3e-3 and ang <= 4.0 and clear:
                break
        if self._gate(e <= 3e-3 and ang <= 4.0 and clear, "align_ik", strict):
            return False

        pre_pos = grasp_at - BACKOFF * a_dir + PRE_LIFT * UP
        q_pre, e, ang, clear = self.solve(pre_pos, -UP, q_align)
        if self._gate(e <= 3e-3 and ang <= 4.0 and clear, "pregrasp_ik", strict):
            return False

        # Come in from above rather than swinging in at glass height: travelling
        # straight to the pre-grasp clips the bowl on the way. Descending behind the
        # glass gives maximum clearance past both the bowl and the foot.
        q_high, _, _, _ = self.solve(np.r_[pre_pos[:2], LIFT_Z], -UP, q_pre)

        # Move in swung slightly off to the side so the open jaws hold a gap from
        # the stem instead of brushing it, then swing back to seat the grip point on
        # the stem once already in position.
        delta = self._swing_away(q_grasp, grasp_at, ENTRY_SWING) - q_grasp[BASE_ROT]

        q_high[JAW] = q_pre[JAW] = q_align[JAW] = q_grasp[JAW] = OPEN
        for q in (q_high, q_pre, q_align):
            q[BASE_ROT] = np.clip(q[BASE_ROT] + delta, self.lb[BASE_ROT], self.ub[BASE_ROT])
        self.move_to(q_high, OPEN, 16, settle=2)
        self.move_to(q_pre, OPEN, 14, settle=4)
        self.move_to(q_align, OPEN, 12, settle=6)
        self.move_to(q_grasp, OPEN, 14, settle=8)
        self.move_to(q_grasp, CLOSE, 12, settle=6)

        # The grasp is rigid from here, so the glass's pose in the tip frame is fixed.
        rel = self._tip_mat(d).T @ (self.glass().xpos - self.grasp_point())
        q_lift, e, _, _ = self.solve(np.r_[grasp_at[:2], LIFT_Z], -UP, q_grasp)
        q_lift[JAW] = CLOSE

        # The flip is q4 += pi. Solving the shelf pose with the roll already applied
        # and arcing through the lift means the glass turns over while being carried.
        q_flip = q_lift.copy()
        q_flip[4] += np.pi if q_lift[4] + np.pi <= self.ub[4] else -np.pi
        q_above, e, clear = self.pose_for_glass(np.r_[place_xy, SHELF_TOP + 0.09], q_flip, rel)
        if self._gate(e <= 5e-3 and clear, "place_ik", strict):
            return False
        q_above[JAW] = CLOSE
        q_rise, _, _, _ = self.solve(np.r_[grasp_at[:2], LIFT_CLEAR], -UP, q_grasp)
        q_rise[JAW] = CLOSE
        self.move_to(q_rise, CLOSE, 16, settle=2)
        self.move_arc(q_lift, q_above, CLOSE, 48, settle=8)
        if self._gate(self.tilt() >= 0.80, "flip", strict):
            return False

        # Seat the foot on the shelf and let it settle before opening, so the glass
        # is set down rather than dropped.
        q_down, _, _ = self.pose_for_glass(np.r_[place_xy, SHELF_TOP + 0.001], q_above, rel)
        q_down[JAW] = CLOSE
        self.move_to(q_down, CLOSE, 24, settle=10)
        self.move_to(q_down, OPEN, 10, settle=6)          # release

        # Only the moving jaw swings open; the fixed one stays against the stem, so
        # pulling straight back drags the glass off the shelf. Swinging the base
        # opens a sideways gap first, then withdraw through it.
        q_clear = q_down.copy()
        q_clear[BASE_ROT] = self._swing_away(q_down, self.glass().xpos, RELEASE_SWING)
        q_clear[JAW] = OPEN
        self.move_to(q_clear, OPEN, 16, settle=4)

        q_back, _, _, _ = self.solve(
            self.grasp_point() - 0.09 * self.approach_dir(q_clear) + 0.05 * UP, UP, q_clear)
        q_back[JAW] = OPEN
        self.move_to(q_back, OPEN, 18)
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


def sample_episode(rng, expert):
    """Glass pose, arm start and place spot. The spawn box is pre-vetted for
    reachability margin, so no rejection sampling is needed."""
    glass_xy = SPAWN_MID + rng.uniform(-SPAWN_HALF, SPAWN_HALF)
    q_init = np.clip(expert.home + rng.normal(0, 0.05, 6), expert.lb, expert.ub)
    q_init[JAW] = OPEN
    place_xy = SHELF_XY + rng.uniform(-0.015, 0.015, 2)
    return glass_xy, q_init, place_xy


_worker: dict = {}


def _init_worker():
    _worker["expert"] = Expert(mujoco.MjModel.from_xml_path(str(SCENE)))


def _episode(seed):
    """One demonstration, or None if it did not end with the glass upright on the
    shelf and nothing but the pads touched. Judged on that outcome rather than on
    the intermediate IK checks: those are conservative and abort runs that go on to
    succeed, which would drop exactly the near-edge spawns the policy needs to see."""
    expert = _worker["expert"]
    glass_xy, q_init, place_xy = sample_episode(np.random.default_rng(seed), expert)
    if not expert.run(glass_xy, q_init, place_xy, strict=False):
        return None
    return expert.frames


def generate(n_episodes=1000, seed=0, workers=None, out_dir=None):
    """Write n_episodes demonstrations to out_dir, resuming if it already holds some.

    Seeds are consumed in order and the next unused one is checkpointed after every
    episode, so an interrupted run picks up exactly where it stopped instead of
    regenerating from scratch. Pass disjoint `seed` ranges to shard across machines
    or across several processes on one machine.
    """
    root = Path(out_dir) if out_dir else REPO_ROOT / f"outputs/so100_flip_{int(time())}"
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
                "observation.image": {"dtype": "video", "shape": (*IMG, 3),
                                      "names": ["height", "width", "channels"]},
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
            for offset, frames in enumerate(pool.imap(_episode, seeds, chunksize=1)):
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
    args = p.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    ds, n = generate(n_episodes=args.episodes, seed=args.seed,
                     workers=args.workers, out_dir=args.out_dir)
    print(f"done: {n} episodes at {ds.root}")
