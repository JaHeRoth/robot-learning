"""Shared training loop for the imitation-learning policies (ACT, DP, FMP, ...).

Everything method-specific about a step lives in the caller's loss_fn, which
receives the raw CPU batch and owns device transfer, normalization,
augmentation, and the loss itself.
"""
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.nn.utils import clip_grad_norm_


def train_loop(
    *,
    model,
    loader,
    opt,
    loss_fn,  # (model, raw CPU batch) -> scalar loss
    num_batches: int,
    out_dir: str | Path,
    sched=None,
    ema_decay: float | None = None,  # None = no EMA
    grad_clip_at: float = 10.0,
    log_every: int = 100,
    checkpoint_every: int = 20_000,
    checkpoint_extra: dict | None = None,  # static entries merged into every checkpoint
    eval_every: int | None = None,
    eval_fn=None,  # (model, step) -> None, called every eval_every steps
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ema_sd = None
    if ema_decay is not None:
        ema_sd = {n: w.clone() for n, w in model.state_dict().items()}

    losses = []
    avg_losses = []
    rolling_avg_loss = 0.0
    step = 1
    while step <= num_batches:
        for batch in loader:
            if step > num_batches:
                break
            loss = loss_fn(model, batch)
            opt.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), grad_clip_at)
            opt.step()
            if sched is not None:
                sched.step()

            if ema_decay is not None:
                model_sd = model.state_dict()
                with torch.no_grad():
                    ema_sd = {
                        n: ema_decay * ema_sd[n] + (1 - ema_decay) * model_sd[n]
                        for n in ema_sd
                    }

            losses.append(loss.item())
            rolling_avg_loss += loss.item() / log_every
            if step % log_every == 0:
                print(f"Step {step}/{num_batches}: Loss={rolling_avg_loss}")
                avg_losses.append(rolling_avg_loss)
                rolling_avg_loss = 0.0

            if eval_fn is not None and step % eval_every == 0:
                model.eval()
                with torch.no_grad():
                    eval_fn(model, step)
                model.train()

            if step % checkpoint_every == 0:
                ckpt = {
                    "model_state": model.state_dict(),
                    "opt_state": opt.state_dict(),
                    **(checkpoint_extra or {}),
                }
                if ema_sd is not None:
                    ckpt["ema_state"] = ema_sd
                torch.save(ckpt, out_dir / f"step_{step:06d}.pt")
            step += 1

    np.save(out_dir / "losses.npy", losses)
    np.save(out_dir / "avg_losses.npy", avg_losses)
    plt.plot(range(log_every, len(avg_losses) * log_every + 1, log_every), avg_losses)
    plt.xlabel("Step")
    plt.ylabel("Training loss")
    plt.yscale("log")
    plt.grid()
    plt.savefig(out_dir / "loss_curve.png", dpi=150, bbox_inches="tight")
    plt.show()
