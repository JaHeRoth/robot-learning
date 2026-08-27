import polars as pl
from lerobot.envs.factory import make_env, make_env_config
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.scripts.eval import rollout

n_seeds = 200
env = make_env(make_env_config("pusht"), n_envs=n_seeds)
horizon = env.call("_max_episode_steps")[0]

names = [
    "act_pusht_baseline",
    "act_pusht_dec7",
    "act_pusht_chunk32",
    "act_pusht_kl1",
    "act_pusht_novae",
    "act_pusht_chunk32_dec7",
    "act_pusht_seed1001",
    "act_pusht_chunk16",
    "act_pusht_seed1002",
    # add "act_pusht_triple", "act_pusht_bs64lr2e5" once the queue finishes
]
checkpoints = ["100000"]

rows = []
for name in names:
    for checkpoint in checkpoints:
        policy = ACTPolicy.from_pretrained(
            f"outputs/train/{name}/checkpoints/{checkpoint}/pretrained_model"
        ).to("cuda")
        for replan_interval in [16, 24, 32]:
            policy.config.n_action_steps = replan_interval
            realizations = rollout(env, policy, seeds=list(range(n_seeds)))
            # Successful episodes contribute 0.95 from their success step through
            # the horizon; failed episodes keep their real rewards throughout.
            succeeded = realizations["success"].any(dim=1, keepdim=True)
            mask = realizations["done"] & succeeded
            imputed_rewards = realizations["reward"] * ~mask + 0.95 * mask
            sum_imputed = (
                imputed_rewards.sum(dim=1)
                + 0.95 * (horizon - imputed_rewards.shape[1])
            )
            rows.extend(
                {
                    "name": name,
                    "checkpoint": checkpoint,
                    "n_action_steps": replan_interval,
                    "seed": seed,
                    "sum_imputed_reward": r,
                    "success": s,
                }
                for seed, (r, s) in enumerate(
                    zip(sum_imputed.tolist(), succeeded.squeeze(1).tolist())
                )
            )

per_episode = pl.DataFrame(rows)
per_episode.write_parquet("outputs/re_eval_per_episode.parquet")

summary = per_episode.group_by(
    ["name", "checkpoint", "n_action_steps"], maintain_order=True
).agg(
    avg_sum_imputed_reward=pl.col("sum_imputed_reward").mean(),
    sem_sum_imputed_reward=pl.col("sum_imputed_reward").std() / n_seeds**0.5,
    success_rate=pl.col("success").mean(),
)
with pl.Config(tbl_rows=-1):
    print(summary)
    with open("outputs/re_eval_summary.txt", "w") as f:
        f.write(str(summary))
