# Log

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
