# Flip

The SO-100 flip task: the arm picks an upside-down wine glass up by its stem, turns it upright and sets it on a shelf. Claude Code wrote everything in this folder under my direction.

There are two versions, with different experts and scenes: `flip` (front camera) and `flip_wristcam` (front and wrist camera). [ablations.md](ablations.md) has the experiments on why ACT learns `flip_wristcam` so much faster.

- `expert.py`: the scripted expert both versions share, and `Version`, which describes a version
- `flip.py`, `wristcam.py`: each version's own motion, episode sampling and scene, and its generator command
- `generate.py`: the generation loop behind those commands
- `env.py`: the eval env for either version

Generate (`so100_flip_100` is seeds from 2,000,000; the wristcam datasets are seeds from 0 with `--randomize-speed`):

    python -m sim.flip.flip --episodes 100 --seed 2000000 --out-dir outputs/so100_flip_100
    python -m sim.flip.wristcam --episodes 1000 --randomize-speed --out-dir outputs/so100_flip2_1k
