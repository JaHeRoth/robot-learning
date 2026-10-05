# Flip

The SO-100 flip task: the arm picks an upside-down wine glass up by its stem, turns it upright and sets it on a shelf. Claude Code wrote everything in this folder under my direction.

There are two versions, with different experts and scenes: `flip` (front camera) and `flip_wristcam` (front and wrist camera). [ablations.md](ablations.md) has the experiments on why ACT learns `flip_wristcam` so much faster.
