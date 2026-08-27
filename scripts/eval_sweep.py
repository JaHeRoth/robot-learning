# %%
from lerobot.envs.factory import make_env, make_env_config
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.scripts.eval import eval_policy
from matplotlib import pyplot as plt

# %%
env = make_env(make_env_config("pusht"), n_envs=50)
policy = (
    ACTPolicy.from_pretrained(
        "outputs/train/act_pusht_baseline/checkpoints/100000/pretrained_model"
    )
    .to("cuda")
)

# %%
replan_intervals = [1, 2, 4, 8, 16, 32, 64, 100]
avg_max_rewards = []
success_rates = []
per_episodes = []
for replan_interval in replan_intervals:
    policy.config.n_action_steps = replan_interval
    info = eval_policy(env, policy, n_episodes=50, start_seed=1000)
    avg_max_rewards.append(info["aggregated"]["avg_max_reward"])
    success_rates.append(info["aggregated"]["pc_success"] / 100)
    per_episodes.append(info["per_episode"])

# %%
plt.plot(replan_intervals, avg_max_rewards, "--o", label="avg_max_reward (0 to 0.95)")
plt.plot(replan_intervals, success_rates, "--o", label="Success rate (0 to 1)")
plt.xlabel("n_action_steps")
plt.xscale("log")
plt.grid()
plt.legend()
plt.savefig("outputs/replan_interval_sweep.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
