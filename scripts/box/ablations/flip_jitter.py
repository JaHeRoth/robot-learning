"""jaheroth/so100_flip_jitter_100: the flip expert with the wristcam expert's trajectory
jitter, and nothing else of it.

Every segment is stretched by an episode-wide speed factor times a per-segment factor,
and the via point of the carry arc is wobbled. Both factors are >= 1, so jitter only
ever slows a segment down. The draws come after the drop-off draw, so every seed keeps
its so100_flip_base100 scene, and the eval env is the base one.

    python -m scripts.box.ablations.flip_jitter --out-dir outputs/flip_jitter_100
"""
from dataclasses import replace

import mujoco
import numpy as np

import scripts.generate_so100_flip_data as G
from scripts.box.ablations.common import generate_cli
from scripts.generate_so100_flip_data import JAW, OPEN, SHELF_XY, SPAWN_HALF, SPAWN_MID
from scripts.tasks import SO100_FLIP

SPEED_RANGE = (1.00, 1.05)  # whole-episode duration factor
SEGMENT_JITTER = (1.00, 1.14)
VIA_WOBBLE = 0.02  # rad of noise on the carry arc's via point


class Expert(G.Expert):
    def dur(self, n_frames):
        """Segment length, stretched by the episode's speed and its own jitter."""
        return max(2, int(round(n_frames * self.speed * self.jitter.uniform(*SEGMENT_JITTER))))

    def wobble(self, q):
        """Noise on a via point only, never on a waypoint the task depends on."""
        out = q.copy()
        out[:5] += self.jitter.normal(0, VIA_WOBBLE, 5)
        return np.clip(out, self.lb, self.ub)

    def move_to(self, q_target, jaw, n_frames, settle=0, record=True):
        """Ramp to a pose over n_frames control frames, then hold for `settle` more.
        The hold matters: the servos lag the ramp, and arriving 14 mm high is the
        difference between grasping the stem and knocking the glass's foot."""
        n_frames = self.dur(n_frames)  # ablation
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

    def move_arc(self, q_via, q_target, jaw, n_frames, settle=0, record=True):
        """Quadratic Bezier through q_via. Interpolating start -> via -> target as
        two straight legs makes the arm finish lifting before it begins turning;
        blending them means the lift, the base swing and the wrist roll all run at
        once, which reads as one motion rather than three."""
        n_frames = self.dur(n_frames)  # ablation
        q_via = self.wobble(q_via)  # ablation
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


def sample_episode(rng, expert):
    """Glass pose, arm start and place spot. The spawn box is pre-vetted for
    reachability margin, so no rejection sampling is needed."""
    glass_xy = SPAWN_MID + rng.uniform(-SPAWN_HALF, SPAWN_HALF)
    q_init = np.clip(expert.home + rng.normal(0, 0.05, 6), expert.lb, expert.ub)
    q_init[JAW] = OPEN
    place_xy = SHELF_XY + rng.uniform(-0.015, 0.015, 2)
    expert.speed = rng.uniform(*SPEED_RANGE)  # ablation
    expert.jitter = np.random.default_rng(rng.integers(2**32))  # ablation
    return glass_xy, q_init, place_xy


G.Expert = Expert
G.sample_episode = sample_episode
G.REPO_ID = "jaheroth/so100_flip_jitter_100"

TASKS = {"so100_flip_jitter_100": replace(SO100_FLIP, dataset_repo_id=G.REPO_ID)}

if __name__ == "__main__":
    generate_cli(G, n_episodes=100)
