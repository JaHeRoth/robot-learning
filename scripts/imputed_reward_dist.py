# %%
import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from scipy import stats

df = pl.read_parquet("outputs/re_eval_per_episode.parquet").filter(
    pl.col("n_action_steps") == 16
)
picks = ["act_pusht_chunk16", "act_pusht_baseline", "act_pusht_dec7", "act_pusht_chunk32_dec7"]
samples = {
    name.removeprefix("act_pusht_"): df.filter(pl.col("name") == name)["sum_imputed_reward"].to_numpy()
    for name in picks
}
rng = np.random.default_rng(0)
B = 10_000

# %%
# Raw per-episode distributions: the shape question (expect success/failure bimodality)
fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
for ax, (name, x) in zip(axes.flat, samples.items()):
    ax.hist(x, bins=30)
    ax.set_title(name)
fig.suptitle("sum_imputed_reward per episode (k=16, seeds 0-199)")
fig.tight_layout()
fig.savefig("outputs/imputed_reward_hists.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Bootstrap distribution of the MEAN: what CIs actually rest on.
boot_means = {
    name: x[rng.integers(len(x), size=(B, len(x)))].mean(axis=1)
    for name, x in samples.items()
}
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, (name, m) in zip(axes.flat, boot_means.items()):
    ax.hist(m, bins=40)
    ax.set_title(name)
fig.suptitle(f"Bootstrap distribution of the mean (B={B}, n=200)")
fig.tight_layout()
fig.savefig("outputs/imputed_reward_boot_hists.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, (name, m) in zip(axes.flat, boot_means.items()):
    stats.probplot(m, dist="norm", plot=ax)
    ax.set_title(name)
fig.suptitle("QQ of bootstrap means vs normal")
fig.tight_layout()
fig.savefig("outputs/imputed_reward_boot_qq.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Endpoint gap between normal-theory and bootstrap-percentile 95% CIs, per n
ns = [10, 25, 50, 100, 200]
fig, ax = plt.subplots(figsize=(7, 5))
for name, x in samples.items():
    gaps = []
    for n in ns:
        boot = x[rng.integers(len(x), size=(B, n))].mean(axis=1)
        normal = x.mean() + 1.96 * x.std(ddof=1) / n**0.5 * np.array([-1, 1])
        percentile = np.percentile(boot, [2.5, 97.5])
        gaps.append(np.abs(normal - percentile).max())
    ax.plot(ns, gaps, "--o", label=name)
ax.set(xscale="log", xlabel="n (episodes)", ylabel="CI endpoint gap (reward)")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.savefig("outputs/imputed_reward_ci_gap_vs_n.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
