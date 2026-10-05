"""The original flip version: front camera only, fixed timing, and a drop-off spot
that varies by up to 15 mm per axis.

    python -m sim.flip.flip --episodes 100 --seed 2000000 --out-dir outputs/so100_flip_100
"""
from pathlib import Path

import mujoco
import numpy as np

from sim.flip.expert import (
    ALIGN_BACK, BACKOFF, BASE_ROT, CLOSE, ENTRY_SWING, FRONT, GRASP_DZ, JAW, LIFT_Z, OPEN,
    PRE_LIFT, RELEASE_SWING, SHELF_TOP, SHELF_XY, SPAWN_HALF, SPAWN_MID, UP, Expert, Version,
)
from sim.flip.generate import parser, run

# Rise this far straight up before any rotation starts. The bowl hangs 0.083 m
# below the grasp point and the roll axis is offset another 0.036, so starting to
# turn any lower swings the bowl into the floor.
LIFT_CLEAR = 0.15


class FlipExpert(Expert):
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

    def run(self, glass_xy, q_init, place_xy, strict=True):
        d = self.d
        self._start(glass_xy, q_init)

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
        q_above, e, clear = self.pose_for_glass(np.r_[place_xy, SHELF_TOP + 0.09], q_flip, rel, UP)
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
        q_down, _, _ = self.pose_for_glass(np.r_[place_xy, SHELF_TOP + 0.001], q_above, rel, UP)
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
        return self._finish(strict)


def sample_episode(rng, expert):
    """Glass pose, arm start and place spot. The spawn box is pre-vetted for
    reachability margin, so no rejection sampling is needed."""
    glass_xy = SPAWN_MID + rng.uniform(-SPAWN_HALF, SPAWN_HALF)
    q_init = np.clip(expert.home + rng.normal(0, 0.05, 6), expert.lb, expert.ub)
    q_init[JAW] = OPEN
    place_xy = SHELF_XY + rng.uniform(-0.015, 0.015, 2)
    return glass_xy, q_init, place_xy


FLIP = Version(
    name="so100_flip",
    scene=Path(__file__).resolve().parent / "scene.xml",
    repo_id="jaheroth/so100_flip",
    cameras=FRONT,
    expert=FlipExpert,
    sample_episode=sample_episode,
    horizon=300,  # demos run 257 frames; leave the policy some slack
)

if __name__ == "__main__":
    run(FLIP, parser().parse_args())
