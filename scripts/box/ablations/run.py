"""Run a main entry point with one ablation's tasks added to scripts.tasks.TASKS.

    python -m scripts.box.ablations.run ABLATION MODULE [ARGS...]
    python -m scripts.box.ablations.run flip_nopj scripts.train --method act \
        --task so100_flip_nopj_100 --chunk-len 50 --n-action-steps 40

Importing an ablation patches the main generator / eval modules, so it is one
ablation per process.
"""
import importlib
import runpy
import sys

from scripts.tasks import TASKS

if __name__ == "__main__":
    ablation = importlib.import_module(f"scripts.box.ablations.{sys.argv[1]}")
    TASKS.update(ablation.TASKS)
    sys.argv = sys.argv[2:]
    runpy.run_module(sys.argv[0], run_name="__main__", alter_sys=True)
