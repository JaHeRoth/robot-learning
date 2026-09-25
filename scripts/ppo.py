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

import torch
from torch import Tensor
from torch.distributions import Categorical
from torch.nn import Module, Sequential, Linear, ReLU
from tqdm import tqdm
from torch.optim import AdamW
from gymnasium.vector import VectorEnv
from torch.nn.utils import clip_grad_norm_

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


class Trajectories:
    def __init__(self, states, actions, rewards, dones, values, logprob):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.dones = dones
        self.values = values  # Length 1 more than all the others
        self.logprob = logprob
        self.advantages = self._build_advantages(gamma=gamma, lambda_=lambda_)  # TODO: Get these from somewhere
        self.returns = self.advantages + self.values[:-1]
        self._flatten()

    def _build_advantages(self, gamma: float, lambda_: float):
        advantages = [None] * len(self.states)
        advantage = 0.0
        for i in reversed(range(len(self.states))):
            advantages[i] = advantage = (
                self.rewards[i] + (1 - self.dones[i].float()) * gamma * self.values[i + 1] - self.values[i]
                + (1 - self.dones[i].float()) * gamma * lambda_ * advantage
            )
            
        return torch.vstack(advantages)

    def _flatten(self):
        # TODO: Flatten dims 0 and 1 of all tensors
        raise NotImplementedError


def gen_trajectories(actor_critic: ActorCritic, envs: VectorEnv, horizon: int) -> Trajectories:
    with torch.no_grad():
        for _ in range(horizon):
            raise NotImplementedError


n_envs = 16
n_cycles = 1000
n_epochs = 3
horizon = 300
mb_size = 64

gamma = 0.99
lambda_ = 0.95
diversity_factor = 0.01
eps = 0.2
lr = 3e-4
weight_decay = 0.0
max_grad_norm = 0.5

envs = TODO
actor_critic = ActorCritic(TODO)  # TODO: Get from envs
opt = AdamW(actor_critic.parameters(), lr=lr, weight_decay=weight_decay)
for _ in tqdm(range(n_cycles)):
    traj = gen_trajectories(actor_critic, envs, horizon)
    minibatch_indices = torch.cat(
        [torch.randperm(len(traj.states)) for _ in range(n_epochs)]
    ).split(mb_size)
    for idx in minibatch_indices:
        batch = traj.slice(idx)
        batch.advantages = (batch.advantages - batch.advantages.mean()) / batch.advantages.std()
        curr_action_dist, curr_values = actor_critic(batch.states)
        curr_logprobs = curr_action_dist.log_prob()
        propensity_weight = (curr_logprobs[batch.actions] - batch.logprob).exp()
        critic_loss = (curr_values - batch.returns).pow(2).mean()
        actor_loss = -(
            torch.minimum(
                propensity_weight * batch.advantages,
                propensity_weight.clip(1 - eps, 1 + eps) * batch.advantages,
            ).mean()
            + diversity_factor * curr_action_dist.entropy(axis=1).mean()
        )
        loss = critic_loss + actor_loss
        opt.zero_grad()
        loss.backward()
        clip_grad_norm_(actor_critic.parameters(), max_grad_norm)
        opt.step()
