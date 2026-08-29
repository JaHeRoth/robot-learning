# Log

## 28 August 2026

- Scaled evals to 5000 held-out seeds per checkpoint; trained 3 seeds of both top configs: batch_size=64 + n_decoder_layers=7 won for every seed
- Reimplemented ACT in plain PyTorch from the paper, catching a discrepancy between paper and reference implementation (decoder positional embeddings)
- Validated my implementation via parameter count (exact match with reference), gradient flow and overfitting a single batch
- Wrote my own training loop on LeRobotDataset, currently training my ACT with the winning hyperparams
- Extended best run 100k→200k steps: success rate 56.4% → 61.4% ± 1.3% (95% CI), more seeds plus an 800k batch_size=8 run queued
- [Posted day-4 update on X](https://x.com/ja_rothschild/status/2093584395475247495) about the discrepancy

## 27 August 2026

- Built a Python eval pipeline: per-episode results to parquet, held-out eval seeds, custom avg_sum_imputed_reward metric
- Ran ~15 training runs tuning one hyperparameter at a time, plus combos and seed replicas
- Best recipe (batch 64, lr 2e-5, 7 decoder layers): 57% ± 4% success (95% CI, 600 held-out episodes), vs ~44% best previously reported for ACT on PushT
- Reimplemented LeRobot's rollout function, bit-exact after two fun determinism bugs
- [Posted day-3 update on X](https://x.com/ja_rothschild/status/2093227915689472365) with the results table

## 26 August 2026

- Explored the PushT dataset
- Trained ACT on PushT on a cloud A10: 0% success with defaults, as others also report for this task
- Swept n_action_steps at eval time: best value (16) gives 24% success, no retraining needed
- [Posted day-2 update on X] with the sweep curve

## 25 August 2026

- Read the ACT paper
- Set up repo + pixi environment (LeRobot, W&B, public runs configured)
- Evaluated pretrained Diffusion Policy on PushT: 6/10 successes, in line with reference
- [Posted first progress update on X](https://x.com/ja_rothschild/status/2092468446705455573) with a rollout clip
