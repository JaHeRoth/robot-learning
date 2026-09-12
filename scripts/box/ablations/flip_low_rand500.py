"""jaheroth/so100_flip_low500 and jaheroth/so100_flip_rand500: two 500-episode subsets of
jaheroth/so100_flip (10k episodes), cut rather than regenerated.

The flip expert's drop-off noise is invisible to the policy, but it can be read back
from the data: forward kinematics at the frame the jaw reopens gives where each
episode let go of the glass. low500 holds the 500 episodes that let go closest to the
median spot (all within 3.8 mm), rand500 a random 500 (11.7 mm off on average). Same
expert, same kind of scenes; only the label noise differs.

    python -m scripts.box.ablations.flip_low_rand500 SO100_FLIP_ROOT OUT_DIR

writes OUT_DIR/so100_flip_low500 and OUT_DIR/so100_flip_rand500, keeping the source
order. SO100_FLIP_ROOT is a local copy of jaheroth/so100_flip, videos included.
"""
import argparse
import json
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd

import scripts.generate_so100_flip_data as G
from scripts.box.merge_lerobot_datasets import merge
from scripts.tasks import SO100_FLIP

N = 500


def release_xy(root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Episode indices, and the tip's xy at the frame the jaw is commanded open again."""
    info = json.loads((root / "meta/info.json").read_text())
    model = mujoco.MjModel.from_xml_path(str(G.SCENE))
    data = mujoco.MjData(model)
    tip = model.site("tip").id
    episodes, xy = [], []
    for ep in range(info["total_episodes"]):
        df = pd.read_parquet(root / info["data_path"].format(
            episode_chunk=ep // info["chunks_size"], episode_index=ep))
        action = np.stack(df["action"].to_numpy())
        state = np.stack(df["observation.state"].to_numpy())
        closed = np.flatnonzero(action[:, G.JAW] < 0.5)
        if len(closed) == 0:
            continue
        reopen = np.flatnonzero(action[closed[0]:, G.JAW] > 0.5)
        release = closed[0] + (reopen[0] if len(reopen) else len(action) - closed[0] - 1)
        data.qpos[:6] = state[min(release, len(state) - 1)]
        mujoco.mj_forward(model, data)
        episodes.append(ep)
        xy.append(data.site_xpos[tip][:2].copy())
    return np.array(episodes), np.array(xy)


TASKS = {
    "so100_flip_low500": replace(SO100_FLIP, dataset_repo_id="jaheroth/so100_flip_low500"),
    "so100_flip_rand500": replace(SO100_FLIP, dataset_repo_id="jaheroth/so100_flip_rand500"),
}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("so100_flip_root", type=Path)
    p.add_argument("out_dir", type=Path)
    args = p.parse_args()

    episodes, xy = release_xy(args.so100_flip_root)
    off = np.linalg.norm(xy - np.median(xy, axis=0), axis=1)
    subsets = {
        "so100_flip_low500": episodes[np.argsort(off)[:N]],
        "so100_flip_rand500": episodes[np.random.default_rng(0).choice(len(episodes), N, replace=False)],
    }
    for name, keep in subsets.items():
        out = args.out_dir / name
        info = merge(out, [args.so100_flip_root], keep={int(e) for e in keep})
        info["repo_id"] = f"jaheroth/{name}"
        (out / "meta/info.json").write_text(json.dumps(info, indent=4))
        print(f"{name}: release {1000 * off[np.isin(episodes, keep)].mean():.2f} mm off on average")
