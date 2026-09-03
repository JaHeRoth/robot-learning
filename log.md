# Log

## 2 September 2026

- Read the Diffusion Policy paper, finding another discrepancy between paper and reference implementation (Eq. 3 adds noise without downscaling the signal; the code uses the standard blend)
- Generated 100 SO-100 reach episodes and trained reference ACT on them; dataset and checkpoints on Hugging Face
- Started reimplementing DP-CNN in plain PyTorch: obs encoder (ResNet-18 with GroupNorm, spatial softmax) and FiLM conditioning done, UNet and sampler next
- Learned: chunk length must be divisible by 4, the reference UNet only uses 2 of its 3 skip connections, 100 denoising steps are enough for actions (image diffusion uses 1000)
- Wrote up the VAE view of ACT, DDPM and DDIM (ELBO gap, small steps, why DDIM can skip), fact-checked against the original papers
- [Posted update on X](https://x.com/ja_rothschild/status/2095399148078141573) with the discrepancy and the writeup

## 1 September 2026

- Finished the MIT flow matching and diffusion course (6.S184)
- Connected DDPM, DDIM and flow matching: wrote a summary of how each reasons from its own perspective, fact-checked against all three papers
- Built a scripted-expert generator for my SO-100 reach task: goal sampling with rejection filtering, 25 Hz control, recording straight into LeRobotDataset format; first episode recorded and verified
- [Posted update on X](https://x.com/ja_rothschild/status/2095036719099679012) with the DDPM/DDIM/FM summary

## 31 August 2026

- Wrapped up PushT: batch_size=8 extended to 1.49M steps; matches batch 64 at equal samples but plateaus ~3 points lower; batch 64 far ahead at equal steps or wall-clock
- Archived 1.02M eval episodes as parquet, published 6 checkpoints to Hugging Face, terminated the cloud box
- Hackerspace had no physical arms; started building a sim env around the SO-100 instead: goal cube, fingertip site, fixed camera, validated with a 3x3 render grid
- More servo experiments: separating torque saturation from genuine low-pass filtering (Lissajous plots)
- [Posted update on X](https://x.com/ja_rothschild/status/2094673837623574535) with the batch-size comparison

## 30 August 2026

- 200k checkpoints per seed: dropping chunk_size=32 wins 2 of 3; best run 66.1% ± 1.3% success (n=5000); 500k extension flat since step 160k
- My from-scratch ACT: 61.1% ± 1.4% at 100k (n=5000), inside the reference seed distribution
- Set up MuJoCo with the SO-ARM100 model: viewer, MJCF read-through, servo physics (droop, deadband, saturation)
- Started the MIT diffusion course as prerequisite for the Diffusion Policy paper
- [Posted update on X](https://x.com/ja_rothschild/status/2094313091588915223) with a video of the arm throwing a fit in sim

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
