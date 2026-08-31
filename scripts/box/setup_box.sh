#!/bin/bash
# Setup for a fresh Lambda GPU instance (tested on A10, Lambda Stack Ubuntu).
# Steps 0-1 happen on the Mac; run the rest ON the instance after first ssh.
#
# 0. LOCAL: add to ~/.ssh/config (replace <IP>):
#      Host a10 <IP>
#          HostName <IP>
#          User ubuntu
#          ForwardAgent yes
#    ForwardAgent lets the box use the Mac's ssh key for GitHub pulls/pushes.
#    Run `ssh-add` locally once per reboot so the agent actually holds the key.
# 1. LOCAL: `scp scripts/setup_box.sh a10:` then `ssh a10 bash setup_box.sh`.
set -euo pipefail

# 2. pixi (installs to ~/.pixi/bin and appends a PATH line to ~/.bashrc)
curl -fsSL https://pixi.sh/install.sh | bash
export PATH="$HOME/.pixi/bin:$PATH"

# 3. git identity + repo (needs the forwarded agent from step 0)
git config --global user.name "Jacob H. Rothschild"
git config --global user.email "20166610+JaHeRoth@users.noreply.github.com"
git clone git@github.com:JaHeRoth/robot-learning.git ~/robot-learning
cd ~/robot-learning

# 4. environment: resolves pixi.toml + pixi.lock (several minutes first time)
pixi install

# 5. HF auth, needed for hub pushes (interactive: paste token when prompted)
pixi run python -c "from huggingface_hub import login; login()"

# 6. sanity checks (last one downloads the pusht dataset to ~/.cache/huggingface)
nvidia-smi
pixi run python -c "import torch; print(torch.cuda.get_device_name(0))"
pixi run python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; print(LeRobotDataset('lerobot/pusht').num_episodes)"

# Notes from the first box's ops history:
# - Nothing was apt-installed beyond the stock image; drivers/CUDA ship with Lambda Stack.
# - Long jobs live in tmux. For anything spawned concurrently, call
#   ~/robot-learning/.pixi/envs/default/bin/{python,lerobot-train} directly:
#   `pixi run` dies silently under concurrent invocation.
# - CPU-only eval workers: CUDA_VISIBLE_DEVICES="" plus torch.set_num_threads(1).
# - Headless rendering if ever needed: MUJOCO_GL=egl.
# - wandb stayed disabled on all extension runs (resume crashes + storage quota);
#   if re-enabling, add --wandb.disable_artifact=true.
