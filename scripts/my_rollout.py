import torch
import numpy as np
from lerobot.envs.factory import make_env, make_env_config
from lerobot.policies.act.modeling_act import ACTPolicy
from gymnasium.vector import VectorEnv


def rollout(
    env: VectorEnv, policy: ACTPolicy, seeds: int | list[int] | None
) -> dict[str, torch.Tensor]:
    assert len(seeds) == env.num_envs
    
    obs, info = env.reset(seed=seeds)
    policy.reset()

    rewards, successes, dones = [], [], []
    done = np.zeros(env.num_envs, dtype=bool)
    while not done.all():
        policy_in = {
            "observation.image": (
                torch.from_numpy(obs["pixels"]).float().to("cuda").permute(0, 3, 1, 2) / 255
            ),
            "observation.state": torch.from_numpy(obs["agent_pos"]).float().to("cuda"),
        }
        with torch.no_grad():
            action: torch.Tensor = policy.selection_action(policy_in)
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
        "reward": torch.tensor(rewards),
        "success": torch.tensor(successes),
        "done": torch.tensor(dones),
    }
        