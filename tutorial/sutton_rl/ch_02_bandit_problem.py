# pylint: disable=invalid-name

# <markdowncell>

# # Chapter 2. Bandit Problems
# In this tutorial, we cover n-Armed Bandit Problem, following Chapter 2 of
# `Reinforcement Learning: An Introduction` to study fundamental properties
# of reinforcement learning, such as evaluative feedback and explotation vs
# exprolitation.

# ### Definition of Bandit Problems

# The following is the definicion/explanation of bandit problem from the book.

# ---

# *You are faced repeatedly with a choice among n different options, or *
# *actions. After each choice you receive a numerical reward chosen from a *
# *stationary probability distribution that depends on the action you *
# *selected. Your objective is to maximize the expected total reward over *
# *some time period, for example, over 1000 action selections, or time steps.*

# *This is the original form of the n-armed bandit problem, so named by *
# *analogy to a slot machine, or "one-armed bandit," except that it has n *
# *levers instead of one. Each action selection is like a play of one of *
# *the slot machine's levers, and the rewards are the payoffs for hitting *
# *the jackpot. Through repeated action selections you are to maximize *
# *your winnings by concentrating your actions on the best levers.*

# ---

# To summarize:
# - You have a set of possible actions.
# - At each time setp, you take action, you receive reward.
# - You can take an action for a certain number of times.
# - Your objective is to maximize *the sum of all rewards* you receive
# through all the trial.
# - Rewards are drawn from distributions correspond to taken actions.
# - The reward distributions are fixed, thus any action taken at any
# time do not have any effect on future rewards.
# - You do not know the reward distributions beforehand.

# You can create bandit problem using `luchador.env.StationaryBandit`
# environment class, as follow.

# <codecell>
from __future__ import division
from __future__ import print_function

import luchador.env
import luchador.agent
import matplotlib.pyplot as plt
import ch_02_bandit_problem_util as util


n_arms = 10
bandit = luchador.env.StationaryBandit(n_arms=n_arms, seed=10)
# `reset` generates rewards distributions with means randomly drawn
# from normal distribution and variance=1
bandit.reset()
# You can peak in the resulting distributions as following.
# This is not know to agent
for i, dist in enumerate(bandit.distributions):
    print('Action {}: {}'.format(i, dist))

# <markdowncell>

# The mean values shown above are called `value` of each action. If agents
# knew these values, then they could solve bandit problem just by selecting
# the action with the highest value. Agents, however can only estimate these
# value.

# ### Exploitation, Exploration and Greedy Action

# Let's estimate action values by taking average rewards.

# <codecell>

# We take each action 3 times just to have estimates for all actions for
# illustration purpose.
n_trials = 3
for i in range(n_arms):
    value = sum([bandit.step(n=i).reward for _ in range(n_trials)]) / n_trials
    print('Action {}: {:7.3f}'.format(i, value))

# <markdowncell>

# We know that these estimates are not accurate, but agent does not know
# the true action values. The action with the highest estimated value is called
# *greedy action*. When making decision on which action to take next,
# agent can eiether repeat the greedy action to maximize total rewards, or
# try other actions. The former is called *exploitation*, and the latter
# is called *exploration*. Since exploitation and exploration cannot be carried
# out at the same time, agents have way to balance actions between them.

# ### Action-value method and epsilon-greedy policy

# Let's incorperate some action selection into simple average action value
# gestimation.

# We consider the following rules.
# 1. Select the actions with highest estimated action values at the time being.
# 2. Behave like 1 most of the time, but every once in a while, select action
# randomly with equal probability.

# Rule 2. is called epsilon-greedy method, where epsilon represents the
# probability of taking random action. Rule 1. is called greedy method but can
# be considered as a special case of epsilon-greedy method, that is epsilon=0.

# To see the different behavior of these rules, let's run an experiment.
# In this experiment, we run 2000 independant 10-armed bandit problem. Agents
# estimate action values by tracking average rewards for each action and select
# the next action based on epsilon-greedy method. We have
# `luchador.agent.EGreedyAgent` class for this.

# <codecell>

epsilons = [0.0, 0.01, 0.1]
mean_rewards, optimal_actions = [], []
for epsilon in epsilons:
    print('Running epsilon = {}...'.format(epsilon))
    env = luchador.env.StationaryBandit(n_arms=10, seed=0)
    agent = luchador.agent.EGreedyAgent(epsilon=epsilon)
    agent.init(env)
    rewards, actions = util.run_episodes(env, agent, episodes=2000, steps=1000)
    mean_rewards.append(rewards)
    optimal_actions.append(actions)

util.plot_result(epsilons, mean_rewards, optimal_actions)
plt.show()
