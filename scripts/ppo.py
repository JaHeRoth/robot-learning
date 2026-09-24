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
