"""Command line shared by the ablation generators."""
import argparse
import os


def generate_cli(G, n_episodes: int, **generate_kwargs):
    """Run G.generate from seed 0, as every ablation dataset was made."""
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True, help="resumed in place if it already exists")
    p.add_argument("--episodes", type=int, default=n_episodes)
    p.add_argument("--workers", type=int, default=None)
    args = p.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    ds, n = G.generate(n_episodes=args.episodes, seed=0, workers=args.workers,
                       out_dir=args.out_dir, **generate_kwargs)
    print(f"done: {n} episodes at {ds.root}")
