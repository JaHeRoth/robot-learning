import torch
import numpy as np
from lerobot.envs.factory import make_env, make_env_config
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.scripts.eval import rollout
from gymnasium.vector import VectorEnv


def my_rollout(
    env: VectorEnv, policy: PreTrainedPolicy, seeds: int | list[int] | None
) -> dict[str, torch.Tensor]:
    assert not isinstance(seeds, list) or len(seeds) == env.num_envs

    device = next(policy.parameters()).device
    obs, info = env.reset(seed=seeds)
    policy.reset()

    rewards, successes, dones = [], [], []
    done = np.zeros(env.num_envs, dtype=bool)
    while not done.all():
        policy_in = {
            "observation.image": (
                (torch.from_numpy(obs["pixels"]).permute(0, 3, 1, 2).contiguous() / 255).to(device)
            ),
            "observation.state": torch.from_numpy(obs["agent_pos"]).float().to(device),
        }
        with torch.no_grad():
            action: torch.Tensor = policy.select_action(policy_in)
        obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
        rewards.append(reward)
        successes.append(
            [
                c is not None and c.get("is_success", False)
                for c in info["final_info"]
            ]
            if "final_info" in info
            else [False] * env.num_envs
        )
        done |= terminated | truncated
        dones.append(done.copy())
    return {
        "reward": torch.from_numpy(np.stack(rewards, axis=1)),
        "success": torch.from_numpy(np.stack(successes, axis=1)),
        "done": torch.from_numpy(np.stack(dones, axis=1)),
    }


def test_my_rollout():
    n_envs = 64
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    env = make_env(make_env_config("pusht"), n_envs=n_envs)
    policy = ACTPolicy.from_pretrained("jaheroth/act_pusht_baseline").to(device)
    seeds = list(range(n_envs))
    expectation = rollout(env, policy, seeds)
    reality = my_rollout(env, policy, seeds)
    for k, v in reality.items():
        assert v.allclose(expectation[k]), f"{k} deviates from expectation"


if __name__ == "__main__":
    test_my_rollout()
    print("PASS")
