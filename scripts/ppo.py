# Repeat the below a bunch of times:
# Roll out a batch tau of trajectories following pi (no grad) for T timesteps, along the way recording states, actions, rewards, done, critic's value estimates, actor's log-probs, plus recording the critic's value for the final state. We get pi(s_t) by passing state tensor into our actor-critic network and sampling from the resulting log-probs.
# Build A_t by walking tau (rewards, done, value estimates, final value estimate) backwards
# For a few steps, sample a minibatch of trajectory/t pairs, each time:
#  Pass state tensor into actor-critic network to get value estimates and log-probs
#  Define loss_critic as mean square of value estimate minus value target (old advantage plus old value estimate)
#  Define rho as prob of selected action divided by old prob of selected action (so exp of difference of log probs)
#  Define loss_actor as mean -min(rho * advantage, clip(rho, 1-eps, 1+eps) * advantage)
#  Define bonus_diversity as mean entropy(log-probs)
#  Define loss as loss_actor + c1 * loss_critic - c2 * bonus_diversity
#  Backpropagate and optimizer step

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.distributions import Categorical
from torch.nn import Module, Sequential, Linear, ReLU
from tqdm import tqdm
from torch.optim import AdamW
import gymnasium
from gymnasium.vector import VectorEnv, SyncVectorEnv
from torch.nn.utils import clip_grad_norm_
from numpy import ndarray

class ActorCritic(Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 50):
        super().__init__()
        self.actor = Sequential(
            Linear(state_dim, hidden_dim), ReLU(), Linear(hidden_dim, action_dim)
        )
        self.critic = Sequential(
            Linear(state_dim, hidden_dim), ReLU(), Linear(hidden_dim, 1)
        )

    def forward(self, state: Tensor) -> tuple[Categorical, Tensor]:
        return Categorical(logits=self.actor(state)), self.critic(state).squeeze(-1)


@dataclass
class Rollout:
    # All have shape (T, n_envs) except values, which has shape (T + 1, n_envs)
    states: Tensor
    actions: Tensor
    rewards: Tensor
    dones: Tensor
    values: Tensor
    logprobs: Tensor

@dataclass
class Transitions:
    # All have shape (B=T*n_envs,)
    states: Tensor
    actions: Tensor
    logprobs: Tensor
    advantages: Tensor
    returns: Tensor

    def __getitem__(self, idx: int | Tensor):
        return Transitions(
            states=self.states[idx],
            actions=self.actions[idx],
            logprobs=self.logprobs[idx],
            advantages=self.advantages[idx],
            returns=self.returns[idx],
        )
    

def build_advantages(rollout: Rollout, gamma: float, lambda_: float):
    advantages = [None] * len(rollout.states)
    advantage = 0.0
    for i in reversed(range(len(rollout.states))):
        advantages[i] = advantage = (
            rollout.rewards[i] + (1 - rollout.dones[i].float()) * gamma * rollout.values[i + 1] - rollout.values[i]
            + (1 - rollout.dones[i].float()) * gamma * lambda_ * advantage
        )
        
    return torch.vstack(advantages)


def build_transitions(rollout: Rollout, gamma: float, lambda_: float):
    advantages = build_advantages(rollout=rollout, gamma=gamma, lambda_=lambda_)
    returns = advantages + rollout.values[:-1]
    return Transitions(
        states=rollout.states.flatten(0, 1),
        actions=rollout.actions.flatten(0, 1),
        logprobs=rollout.logprobs.flatten(0, 1),
        advantages=advantages.flatten(0, 1),
        returns=returns.flatten(0, 1),
    )


def sim_rollout(actor_critic: ActorCritic, envs: VectorEnv, horizon: int, start_state: ndarray) -> tuple[Rollout, ndarray]:
    states, actions, rewards, dones, values, logprobs = [], [], [], [], [], []
    obs = start_state
    with torch.no_grad():
        for _ in range(horizon):
            states.append(torch.tensor(obs, dtype=torch.float32))
            action_dist, value = actor_critic(states[-1])
            action = action_dist.sample()
            obs, reward, terminated, truncated, info = envs.step(action.numpy())
            actions.append(action)
            rewards.append(torch.tensor(reward, dtype=torch.float32))
            dones.append(torch.tensor(terminated | truncated))
            values.append(value)
            logprobs.append(action_dist.log_prob(action))
        values.append(actor_critic(torch.tensor(obs, dtype=torch.float32))[1])
    return (
        Rollout(
            states=torch.stack(states),
            actions=torch.stack(actions),
            rewards=torch.stack(rewards),
            dones=torch.stack(dones),
            values=torch.stack(values),
            logprobs=torch.stack(logprobs),
        ), obs
    )


n_envs = 16
n_cycles = 1000
n_epochs = 3
horizon = 300
batch_size = 64

gamma = 0.99
lambda_ = 0.95
diversity_factor = 0.01
eps = 0.2
lr = 3e-4
weight_decay = 0.0
max_grad_norm = 0.5

envs = SyncVectorEnv(
    [
        lambda: gymnasium.make("CartPole-v1")
        for _ in range(n_envs)
    ]
)
start_state, info = envs.reset(seed=list(range(envs.num_envs)))
actor_critic = ActorCritic(
    state_dim=envs.single_observation_space.shape[0],
    action_dim=envs.single_action_space.n,
)
opt = AdamW(actor_critic.parameters(), lr=lr, weight_decay=weight_decay)
for _ in tqdm(range(n_cycles)):
    rollout, start_state = sim_rollout(actor_critic, envs, horizon, start_state)
    transitions = build_transitions(rollout=rollout, gamma=gamma, lambda_=lambda_)
    batch_indices = torch.cat(
        [torch.randperm(len(transitions.states)) for _ in range(n_epochs)]
    ).split(batch_size)
    for idx in batch_indices:
        batch = transitions[idx]
        batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)
        curr_action_dist, curr_values = actor_critic(batch.states)
        curr_logprobs = curr_action_dist.log_prob(batch.actions)  # Only for executed actions, so shape=(B,)
        propensity_weight = (curr_logprobs - batch.logprobs).exp()
        critic_loss = (curr_values - batch.returns).pow(2).mean()
        actor_loss = -(
            torch.minimum(
                propensity_weight * batch.advantages,
                propensity_weight.clip(1 - eps, 1 + eps) * batch.advantages,
            ).mean()
            + diversity_factor * curr_action_dist.entropy().mean()
        )
        loss = critic_loss + actor_loss
        opt.zero_grad()
        loss.backward()
        clip_grad_norm_(actor_critic.parameters(), max_grad_norm)
        opt.step()
