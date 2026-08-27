# Log

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
