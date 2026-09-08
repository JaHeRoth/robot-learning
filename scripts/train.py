"""Single entrypoint for training any policy on any task.

    python -m scripts.train --method act --task so100_reach --chunk-len 50
"""
import argparse

from scripts.tasks import TASKS
from scripts.train_act import train_act
from scripts.train_dp import LossType, train_dp

if __name__ == "__main__":  # Guard is required: AsyncVectorEnv re-imports this via spawn
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["act", *[t.value for t in LossType]], required=True)
    p.add_argument("--task", choices=TASKS, default="pusht")
    p.add_argument("--seed", type=int, default=0)
    # None = the method's own default
    p.add_argument("--chunk-len", type=int)
    p.add_argument("--n-action-steps", type=int)
    args = p.parse_args()

    kwargs = {"task": TASKS[args.task], "seed": args.seed}
    for name in ["chunk_len", "n_action_steps"]:
        if getattr(args, name) is not None:
            kwargs[name] = getattr(args, name)

    if args.method == "act":
        train_act(**kwargs)
    else:
        train_dp(loss_type=LossType(args.method), **kwargs)
