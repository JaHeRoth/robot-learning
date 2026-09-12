"""jaheroth/so100_flip_roll100: the flip expert with the wrist pre-rolled half a turn
before the grasp, as the wristcam expert does, so the flip during the carry turns
against the base swing instead of with it.

The roll axis is the approach axis, so a half turn leaves the approach untouched but
points the grasped stem the other way: every stem direction the expert solves for is
flipped, -UP to UP before the flip and UP to -UP after it. Timing, gates, waypoint
geometry, scenes and the eval env are those of the flip task.

    python -m scripts.box.ablations.flip_roll --out-dir outputs/flip_roll100
"""
from dataclasses import replace

import mujoco
import numpy as np

import scripts.generate_so100_flip_data as G
from scripts.box.ablations.common import generate_cli
from scripts.generate_so100_flip_data import (
    ALIGN_BACK, BACKOFF, BASE_ROT, CLOSE, ENTRY_SWING, GRASP_DZ, JAW, LIFT_CLEAR, LIFT_Z,
    OPEN, PRE_LIFT, RELEASE_SWING, SHELF_TOP, SHELF_XY, UP,
)
from scripts.tasks import SO100_FLIP


class Expert(G.Expert):
    def pose_for_glass(self, target, q_seed, rel):
        """Arm pose that puts the *glass* at target, upright. The grasp is rigid, so
        the glass sits at grasp_point + R_tip @ rel; R_tip depends on the pose being
        solved, so iterate. Lets the shelf poses be computed before the flip has
        physically happened, which is what allows rolling and carrying at once."""
        q, e, clear = q_seed, np.inf, False
        for _ in range(3):
            R = self._tip_mat(self._load(q))
            q, e, _, clear = self.solve(target - R @ rel, -UP, q)  # ablation: was UP
        return q, e, clear

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
        # ablation: solving the pre-rolled branch straight from the home seed does not
        # converge, and rolling a solved pose drags the grip point off the stem (the
        # roll axis sits ~3.6 cm from the jaw midpoint). So solve the usual branch,
        # roll it half a turn into a seed, and re-solve from there.
        q_easy, _, _, _ = self.solve(grasp_at, -UP, self.seed(glass_xy))
        seed_roll = q_easy.copy()
        step = np.pi if q_easy[4] + np.pi <= self.ub[4] else -np.pi
        seed_roll[4] = np.clip(q_easy[4] + step, self.lb[4], self.ub[4])
        q_grasp, e, ang, clear = self.solve(grasp_at, UP, seed_roll)
        if self._gate(e <= 3e-3 and ang <= 4.0 and clear, "grasp_ik", strict):
            return False

        # Two waypoints, not one. The last leg must be purely horizontal so the jaw
        # slides in under the foot; descending diagonally onto the stem sweeps the
        # gripper's upper body straight through the foot's overhang.
        a_dir = self.approach_dir(q_grasp)
        for back in (ALIGN_BACK, 0.025, 0.02):
            q_align, e, ang, clear = self.solve(grasp_at - back * a_dir, UP, q_grasp)  # ablation: was -UP
            if e <= 3e-3 and ang <= 4.0 and clear:
                break
        if self._gate(e <= 3e-3 and ang <= 4.0 and clear, "align_ik", strict):
            return False

        pre_pos = grasp_at - BACKOFF * a_dir + PRE_LIFT * UP
        q_pre, e, ang, clear = self.solve(pre_pos, UP, q_align)  # ablation: was -UP
        if self._gate(e <= 3e-3 and ang <= 4.0 and clear, "pregrasp_ik", strict):
            return False

        # Come in from above rather than swinging in at glass height: travelling
        # straight to the pre-grasp clips the bowl on the way. Descending behind the
        # glass gives maximum clearance past both the bowl and the foot.
        q_high, _, _, _ = self.solve(np.r_[pre_pos[:2], LIFT_Z], UP, q_pre)  # ablation: was -UP

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
        q_lift, e, _, _ = self.solve(np.r_[grasp_at[:2], LIFT_Z], UP, q_grasp)  # ablation: was -UP
        q_lift[JAW] = CLOSE

        # The flip is q4 += pi. Solving the shelf pose with the roll already applied
        # and arcing through the lift means the glass turns over while being carried.
        q_flip = q_lift.copy()
        q_flip[4] += np.pi if q_lift[4] + np.pi <= self.ub[4] else -np.pi
        q_above, e, clear = self.pose_for_glass(np.r_[place_xy, SHELF_TOP + 0.09], q_flip, rel)
        if self._gate(e <= 5e-3 and clear, "place_ik", strict):
            return False
        q_above[JAW] = CLOSE
        q_rise, _, _, _ = self.solve(np.r_[grasp_at[:2], LIFT_CLEAR], UP, q_grasp)  # ablation: was -UP
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
            self.grasp_point() - 0.09 * self.approach_dir(q_clear) + 0.05 * UP, -UP, q_clear)  # ablation: was UP
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


G.Expert = Expert
G.REPO_ID = "jaheroth/so100_flip_roll100"

TASKS = {"so100_flip_roll100": replace(SO100_FLIP, dataset_repo_id=G.REPO_ID)}

if __name__ == "__main__":
    generate_cli(G, n_episodes=100)
