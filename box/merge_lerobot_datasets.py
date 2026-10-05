"""Concatenate several LeRobotDataset directories into one.

Generating in one process leaves cores idle, so the way to go faster is to run
several generators over disjoint seed ranges and merge the shards afterwards.
Episode indices and the global frame index are renumbered; per-episode stats and
videos are carried over as-is.

    python -m box.merge_lerobot_datasets OUT SHARD [SHARD ...]
"""
import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def _read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def merge(out_dir: Path, shards: list[Path], limit: int | None = None,
          keep: set[int] | None = None) -> dict:
    info = json.loads((shards[0] / "meta/info.json").read_text())
    chunks_size = info["chunks_size"]
    video_keys = [k for k, v in info["features"].items() if v["dtype"] == "video"]

    out_dir.mkdir(parents=True, exist_ok=True)
    episodes, ep_stats, tasks = [], [], {}
    new_ep = global_index = 0

    for shard in shards:
        s_info = json.loads((shard / "meta/info.json").read_text())
        if s_info["features"] != info["features"] or s_info["fps"] != info["fps"]:
            raise ValueError(f"{shard} has different features/fps than {shards[0]}")
        # Task strings are the unit of identity; per-shard indices need remapping.
        s_tasks = {t["task_index"]: t["task"] for t in _read_jsonl(shard / "meta/tasks.jsonl")}
        for task in s_tasks.values():
            tasks.setdefault(task, len(tasks))
        s_eps = {e["episode_index"]: e for e in _read_jsonl(shard / "meta/episodes.jsonl")}
        s_stats = {e["episode_index"]: e for e in _read_jsonl(shard / "meta/episodes_stats.jsonl")}

        for old_ep in sorted(s_eps):
            if limit is not None and new_ep >= limit:
                break
            if keep is not None and old_ep not in keep:
                continue
            src = shard / s_info["data_path"].format(
                episode_chunk=old_ep // chunks_size, episode_index=old_ep)
            df = pd.read_parquet(src)
            df["episode_index"] = new_ep
            df["index"] = range(global_index, global_index + len(df))
            df["task_index"] = df["task_index"].map(lambda i: tasks[s_tasks[i]])

            dst = out_dir / info["data_path"].format(
                episode_chunk=new_ep // chunks_size, episode_index=new_ep)
            dst.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(dst, index=False)

            for key in video_keys:
                v_src = shard / s_info["video_path"].format(
                    episode_chunk=old_ep // chunks_size, video_key=key, episode_index=old_ep)
                v_dst = out_dir / info["video_path"].format(
                    episode_chunk=new_ep // chunks_size, video_key=key, episode_index=new_ep)
                v_dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(v_src, v_dst)

            episodes.append({**s_eps[old_ep], "episode_index": new_ep})
            ep_stats.append({**s_stats[old_ep], "episode_index": new_ep})
            global_index += len(df)
            new_ep += 1
        print(f"  {shard.name}: +{len(s_eps)} episodes (total {new_ep})")

    _write_jsonl(out_dir / "meta/episodes.jsonl", episodes)
    _write_jsonl(out_dir / "meta/episodes_stats.jsonl", ep_stats)
    _write_jsonl(out_dir / "meta/tasks.jsonl",
                 [{"task_index": i, "task": t} for t, i in tasks.items()])
    info.update(
        total_episodes=new_ep,
        total_frames=global_index,
        total_videos=new_ep * len(video_keys),
        total_tasks=len(tasks),
        total_chunks=(new_ep + chunks_size - 1) // chunks_size,
        splits={"train": f"0:{new_ep}"},
    )
    (out_dir / "meta/info.json").write_text(json.dumps(info, indent=4))
    print(f"merged {new_ep} episodes / {global_index} frames -> {out_dir}")
    return info


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out_dir", type=Path)
    p.add_argument("shards", type=Path, nargs="+")
    p.add_argument("--limit", type=int, default=None, help="stop after this many episodes")
    p.add_argument("--keep", type=Path, default=None,
                   help="JSON file holding a list of source episode indices to keep")
    args = p.parse_args()
    keep = set(json.loads(args.keep.read_text())) if args.keep else None
    merge(args.out_dir, args.shards, args.limit, keep)
