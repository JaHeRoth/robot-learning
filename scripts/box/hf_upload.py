"""Upload curated checkpoints + my_act to HF Hub as public model repos."""
from pathlib import Path

from huggingface_hub import HfApi

api = HfApi()
GH = "https://github.com/JaHeRoth/robot-learning"
CARD = """---
license: apache-2.0
tags: [robotics, act, pusht, lerobot]
---
# {title}

{desc}

Part of a 6-week robot-learning training block ({gh}). Evaluated on gym-pusht at
n_action_steps=16; avg_sum_imputed_reward imputes 0.95/step to horizon 300 after success.
"""
REPOS = [
    ("act_pusht_bs64_dec7_seed1002_160k",
     "outputs/train/act_pusht_bs64_dec7_seed1002_500k/checkpoints/160000/pretrained_model",
     "Best tuned ACT on PushT: 164.4 avg_sum_imputed_reward, 66.7% success (n=5000 seeds). "
     "LeRobot ACT, batch 64, lr 2e-5, 7 decoder layers, chunk 100, kl_weight 10, 160k steps."),
    ("act_pusht_bs64_dec7_100k",
     "outputs/train/act_pusht_bs64_dec7/checkpoints/100000/pretrained_model",
     "The original tuned champion recipe at 100k steps: 154.4 avg_sum_imputed_reward, 56.4% success (n=5000)."),
    ("act_pusht_dec7_bs8_800k",
     "outputs/train/act_pusht_dec7_800k/checkpoints/800000/pretrained_model",
     "Batch-8 equal-sample-budget run (800k steps = 6.4M samples): 159.1 avg_sum_imputed_reward, "
     "60.2% success (n=5000). Matches batch-64 at 100k steps to one decimal."),
    ("act_pusht_baseline_100k",
     "outputs/train/act_pusht_baseline/checkpoints/100000/pretrained_model",
     "Stock LeRobot ACT config on PushT: ~0% success with default inference settings; "
     "the starting point of the tuning journey."),
    ("my_act_pusht",
     "outputs/my_act",
     "ACT reimplemented from scratch in PyTorch (see scripts/act.py + scripts/my_train.py in the repo): "
     "157.1 avg_sum_imputed_reward, 61.1% success at 100k steps (n=5000) - inside the reference "
     "implementation seed distribution. Raw state_dicts every 20k steps + loss curves."),
]
for name, folder, desc in REPOS:
    repo = f"jaheroth/{name}"
    api.create_repo(repo, repo_type="model", exist_ok=True)
    api.upload_folder(repo_id=repo, folder_path=folder, commit_message="upload from training box")
    card = CARD.format(title=name, desc=desc, gh=GH)
    api.upload_file(repo_id=repo, path_in_repo="README.md", path_or_fileobj=card.encode(),
                    commit_message="model card")
    print("uploaded", repo, flush=True)
print("ALL UPLOADS DONE")
