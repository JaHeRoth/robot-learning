import json
from lerobot.envs.factory import make_env, make_env_config
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.scripts.eval import eval_policy

env = make_env(make_env_config("pusht"), n_envs=50)
runs = ["act_pusht_baseline", "act_pusht_dec7", "act_pusht_chunk32", "act_pusht_kl1",
        "act_pusht_novae", "act_pusht_chunk32_dec7", "act_pusht_seed1001",
        "act_pusht_chunk16", "act_pusht_seed1002"]
rows = {}
for r in runs:
    p = ACTPolicy.from_pretrained(f"outputs/train/{r}/checkpoints/last/pretrained_model").to("cuda")
    p.config.n_action_steps = 16
    a = eval_policy(env, p, n_episodes=50, start_seed=1000)["aggregated"]
    rows[r] = a
    print(r, round(a["avg_max_reward"], 3), a["pc_success"], flush=True)
json.dump(rows, open("outputs/final_ckpt_metrics.json", "w"), indent=2)
