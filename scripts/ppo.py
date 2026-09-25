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
from torch.nn import Module, Sequential, Linear, ReLU, Softmax
from tqdm import tqdm
from torch.optim import AdamW
from gymnasium import VectorEnv

class ActorCritic(Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 50):
        super().__init__()
        self.actor = Sequential(
            Linear(state_dim, hidden_dim), ReLU(), Linear(hidden_dim, action_dim), Softmax()
        )
        self.critic = Sequential(
            Linear(state_dim, hidden_dim), ReLU(), Linear(hidden_dim, 1)
        )

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor]:
        return self.actor(state), self.critic(state)


class Trajectories:
    def __init__(self, states, actions, rewards, dones, values, logprobs, final_val):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.dones = dones
        self.values = values  # Length 1 more than all the others
        self.logprobs = logprobs
        self.final_val = final_val
        self.advantages = self.build_advantages_(gamma=gamma, lambda_=lambda_)  # TODO: Get these from somewhere

    def build_advantages_(self, gamma: float, lambda_: float):
        advantages = [None] * len(self.states)
        for i in reversed(range(len(self.states))):
            advantages[i] = (
                self.rewards[i] + (1 - self.dones[i]) * gamma * self.values[i + 1] - self.values[i]
                + (1 - self.dones[i]) * gamma * lambda_ * advantages[i + 1]
            )
        return torch.vstack(advantages)


def gen_trajectories(actor_critic: ActorCritic, envs: VectorEnv, horizon: int) -> Trajectories:
    with torch.no_grad():
        for _ in range(horizon):
            raise NotImplementedError


n_envs = 16
n_cycles = 1000
n_steps = 100
horizon = 300

gamma = 0.99
lambda_ = 0.95
diversity_factor = 0.01
eps = 0.2

envs = TODO
actor_critic = ActorCritic(TODO)  # TODO: Get from envs
opt = AdamW(actor_critic.parameters())
for _ in tqdm(range(n_cycles)):
    traj = gen_trajectories(actor_critic, envs, horizon)
    batch_permutes = None
    for _ in range(n_steps):
        if batch_permutes is None:
            batch_permutes = torch.randperm(len(traj.states))
        batch = traj.slice(batch_permutes.next())
        batch.advantages = (batch.advantages - batch.advantages.mean()) / batch.advantages.std()
        curr_logprobs, curr_values = actor_critic(batch.states)
        propensity_weight = (curr_logprobs[batch.actions] - batch.logprobs[batch.actions]).exp()
        critic_loss = (curr_values - (batch.advantages + batch.values)).pow(2).mean()
        actor_loss = -(
            torch.minimum(
                propensity_weight * batch.advantages,
                propensity_weight.clip(1 - eps, 1 + eps) * batch.advantages,
            ).mean()
            + diversity_factor * curr_logprobs.entropy(axis=1).mean()
        )
        loss = critic_loss + actor_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
