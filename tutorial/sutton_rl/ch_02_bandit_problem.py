"""Tutorial for Bandit problem, following Chapter 2 of Sutton"""
# <markdowncell>

# # Chapter 2. Bandit Problems
# In this tutorial, we cover $n$-Armed Bandit Problem, following Chapter 2 of
# `Reinforcement Learning: An Introduction` to study fundamental properties
# of reinforcement learning, such as evaluative feedback and explotation vs
# exprolitation.

# <markdowncell>

# ### Definition of Bandit Problems

# The following is the definition/explanation of bandit problem from the book.

# ---

# *You are faced repeatedly with a choice among $n$ different options, or *
# *actions. After each choice you receive a numerical reward chosen from a *
# *stationary probability distribution that depends on the action you *
# *selected. Your objective is to maximize the expected total reward over *
# *some time period, for example, over 1000 action selections, or time steps.*

# *This is the original form of the $n$-armed bandit problem, so named by *
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

# <markdowncell>

# You can create bandit problem using by subclassing luchador base environment
# class, as follow.

# <codecell>

# pylint: disable=invalid-name,attribute-defined-outside-init
# pylint: disable=redefined-outer-name
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

import luchador.env
import luchador.agent

import ch_02_bandit_problem_util as util


class Bandit(luchador.env.BaseEnvironment):
    """N-armed bandit problem

    Parameters
    ----------
    n_arms : int
        The number of arms of bandit
    """
    def __init__(self, n_arms, seed=None):
        self.n_arms = n_arms
        self.rng_sample = np.random.RandomState(seed=seed)

    @property
    def n_actions(self):
        return self.n_arms

    def reset(self):
        """Set distribution mean randomly"""
        self.mean = self.rng_sample.randn(self.n_arms)
        self.stddev = np.ones(self.n_arms)
        return luchador.env.Outcome(
            reward=0, observation=None, terminal=False)

    def step(self, n):
        """Sample reward from the given distribution"""
        reward = self.rng_sample.normal(loc=self.mean[n], scale=self.stddev[n])
        return luchador.env.Outcome(
            reward=reward, observation=None, terminal=False)

    def __str__(self):
        return '\n'.join(
            'Acttion {:3d}: Mean {:5.2f}, Stddev {:5.2f}'.format(
                i, self.mean[i], self.stddev[i]) for i in range(self.n_arms)
        )


# <markdowncell>

# You need to initialize reward distribution as follow.

# <codecell>

bandit = Bandit(n_arms=10, seed=10)
# `reset` generates rewards distributions with means randomly drawn
# from normal distribution and variance=1
bandit.reset()

# <markdowncell>

# You can peak in the resulting distributions as following.
# This is not know to agent

# <codecell>

print(bandit)

# <markdowncell>

# The mean values shown above are called `value` of each action. If agents
# knew these values, then they could solve bandit problem just by selecting
# the action with the highest value. Agents, however can only estimate these
# value.

# <markdowncell>

# You can use `step` method to take an action over the environment.
# The argument to the method in this case is the index of distribution from
# which reward is drawn.

# <codecell>

for action in range(bandit.n_actions):
    outcome = bandit.step(action)
    print(action, '{:8.3f}'.format(outcome.reward))

# <markdowncell>

# ### Exploitation, Exploration and Greedy Action

# Let's estimate action values by taking average rewards.

# <codecell>

# We take each action 3 times just to have estimates for all actions for
# illustration purpose.
n_trials = 3
print('Hypothetical Estimation:')
for i in range(bandit.n_actions):
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

# <markdowncell>

# ### Action-value method and $\epsilon$-greedy policy

# Let's incorperate some action selection into simple average action value
# gestimation.

# We consider the following rules.
# 1. Select the actions with highest estimated action values at the time being.
# 2. Behave like 1 most of the time, but every once in a while, select action
# randomly with equal probability.

# Rule 2. is called $\epsilon$-greedy method, where $\epsilon$ represents the
# probability of taking random action. Rule 1. is called greedy method but can
# be considered as a special case of $\epsilon$-greedy method, that is
# $\epsilon$=0.

# To see the different behavior of these rules, let's run an experiment.
# In this experiment, we run 2000 independant 10-armed bandit problem.

# Agents estimate action values by tracking average rewards for each action
# and select the next action based on $\epsilon$-greedy method.
# Such agent can be implemented as following, using base luchador agent class.

# <codecell>

class EGreedyAgent(luchador.agent.BaseAgent):
    """Simple E-Greedy policy for stationary environment

    Parameters
    ----------
    epsolon : float
        The probability to take random action.

    step_size : 'average' or float
        Parameter to adjust how action value is estimated from the series of
        observations. When 'average', estimated action value is simply the mean
        of all the observed rewards for the action. When float, estimation is
        updated with weighted sum over current estimation and newly observed
        reward value.

    initial_q : float
        Initial Q value for all actions

    seed : int
        Random seed
    """
    def __init__(self, epsilon, step_size='average', initial_q=0.0, seed=None):
        self.epsilon = epsilon
        self.step_size = step_size
        self.initial_q = initial_q
        self.rng = np.random.RandomState(seed=seed)

    def reset(self, observation):
        self.q_values = [self.initial_q] * self.n_actions
        self.n_trials = [self.initial_q] * self.n_actions

    def init(self, environment):
        self.n_actions = environment.n_actions

    def observe(self, action, outcome):
        """Update the action value estimation based on observed outcome"""
        r, n, q = outcome.reward, self.n_trials[action], self.q_values[action]
        alpha = 1 / (n + 1) if self.step_size == 'average' else self.step_size
        self.q_values[action] += (r - q) * alpha
        self.n_trials[action] += 1

    def act(self):
        """Choose action based on e-greedy policy"""
        if self.rng.rand() < self.epsilon:
            return self.rng.randint(self.n_actions)
        else:
            return np.argmax(self.q_values)


# <markdowncell>

# Run 10-armed test bed for with different $\epsilon$-greedy method, and
# plot rewards and the number of optimal actions taken.

# <codecell>

epsilons = [0.0, 0.01, 0.1]
mean_rewards, optimal_actions = [], []
for eps in epsilons:
    print('Running epsilon = {}...'.format(eps))
    env = Bandit(n_arms=10, seed=0)
    agent = EGreedyAgent(epsilon=eps)
    agent.init(env)
    rewards, actions = util.run_episodes(env, agent, episodes=2000, steps=1000)
    mean_rewards.append(rewards)
    optimal_actions.append(actions)

util.plot_epsilon_comparison(epsilons, mean_rewards, optimal_actions)
plt.show()

# <markdowncell>

# ### Imcremental Implementation to non-stationary extention
# $Q_t$, estimated value of an action at time step $t$, as mean
# observed reward can be written in recursive manner as follow
#
# \begin{align}
# Q_k &= \frac{R_1 +R_2 + \dots +R_{K_a}}{K_a} \\
#     &= \frac{1}{K_a} \sum_{i=1}^{K_a}R_{i} \\
#     &= \frac{1}{K_a} ( R_{K_a} + \sum_{i=1}^{K_a - 1}R_{i} ) \\
#     &= \frac{1}{K_a}
#          \left\{
#            R_{K_a} + ( K_a - 1 ) \frac{1}{K_a - 1}\sum_{i=1}^{K_a - 1}R_{i}
#          \right\} \\
#     &= \frac{1}{K_a} \left\{ R_{K_a} + ( K_a - 1 ) Q_{k-1} \right\} \\
#     &= Q_{k-1} + \frac{1}{K_a} ( R_{K_a} - Q_{k-1} )
# \end{align}
#
# Where $K_a$ denotest the number of action $a$ was taken.
#
# When computing mean value, we need to keep the list of received rewards in
# non-iterative form. But using iterative form, all we need to store is the
# latest estimations. Not only this is memory-efficient, but also, this form
# enables us to extend estimation to non-stationary problem.
#
# The form above can be generalized into update expression
#
# $$ NewEstimate \leftarrow OldEstimate + StepSize ( Target - OldEstimate ) $$
#
# The expression $ ( Target - OldEstimate ) $ is an $ error $ in the estimate.
# It is reduced by taking a step toward the $Target$.
# The target is presumed to indicate a desirable direction in which to move,
# though it may be noisy. In the case above, for example, the target is the
# $k$-th reward.

# <markdowncell>

# ### Nonstationary Problem
#
# In the original mean reward value computation, $StepSize$ parameter, (denoted
# $\alpha$ from now on) changes from time step to time step.
#
# $$ \alpha = \frac{1}{K_{a}} $$
#
# The influence of $ \alpha $ becomes clearer by taking iterative form.
#
# \begin{align}
# Q_{k+1}
#   &= Q_{k} +
#      \alpha ( R_{k} - Q_{k} ) \\
#   &= \alpha R_{k} +
#      (1 - \alpha) Q_{k} \\
#   &= \alpha R_{k} +
#      (1 - \alpha) \left\{
#        \alpha R_{k-1} + (1 - \alpha) Q_{k-1}
#      \right\} \\
#   &= \alpha R_{k} +
#      \alpha (1 - \alpha) R_{k-1} +
#      (1 - \alpha)^{2} Q_{k-1} \\
#   &= \alpha R_{k} +
#      \alpha (1 - \alpha) R_{k-1} +
#      (1 - \alpha)^{2} \left\{
#        \alpha R_{k-2} + (1 - \alpha) Q_{k-2}
#      \right\} \\
#   &= \alpha R_{k} +
#      \alpha (1 - \alpha) R_{k-1} +
#      \alpha (1 - \alpha)^2 R_{k-2} +
#      (1 - \alpha)^{3} Q_{k-2} \\
#   &= \alpha R_{k} +
#      \alpha (1 - \alpha)       R_{k-1} +
#      \alpha (1 - \alpha)^{2}   R_{k-2} +
#      \dots +
#      \alpha (1 - \alpha)^{k-1} R_{1} +
#      (1 - \alpha)^{k} Q_1 \\
#   &= (1 - \alpha)^{k} Q_1 +
#      \sum_{i=1}^{k}
#        \alpha ( 1 - \alpha) ^ {k-i} R_{i}
# \end{align}
#
# For $ 0 \lt \alpha \lt 1 $,
# $ (1 - \alpha) + \sum_{i=1}^{k} \alpha ( 1 - \alpha) ^ {k-i} = 1 $ holds.
# Therefore, the estimation at $ k+1 $ step is weighted average of initial
# estimation $ Q_1 $ and all rewards received.
#
# By using constants value for $\alpha$, the influence from past rewards decay
# exponentially and new rewards have stronger effect on estimation. This is
# more suitable in case the problem is not stationary.

# To see the imapct of $ StepSize $ over bahavior, we can modify the `Bandit`
# class to transition at each time step. (Exercise 2.6)

# <codecell>


class RandomWalkBandit(Bandit):
    """Bandit with true action values taking independent random walks"""
    def __init__(self, n_arms, seed=None):
        super(RandomWalkBandit, self).__init__(n_arms, seed)

        # For separating the effect of random walk from and random sampling
        self.rng_random_walk = np.random.RandomState(seed=0)

    def reset(self):
        outcome = super(RandomWalkBandit, self).reset()
        self.mean[:] = 0.000001 * self.rng_sample.randn(self.n_arms)
        # We could also do `self.mean[:] = 0` but that will cause
        # optimal action ratio to be 100 % at the beginnig, which is not
        # our intention.
        return outcome

    def step(self, n):
        outcome = super(RandomWalkBandit, self).step(n)
        # Each time step, action value distribution takes random walk
        self.mean[:] += self.rng_random_walk.randn(self.n_arms)
        return outcome

# <markdowncell>

# In `RandomWalkBandit` class, reward distribution at the beginning are no
# different each other, but each distribution takes random walk at each time
# step.

# We can run the same procedure as in above.

# <codecell>


epsilon = 0.1
step_sizes = ['average', 0.1]
mean_rewards, optimal_actions = [], []
for step_size in step_sizes:
    print('Running step_size = {}...'.format(step_size))
    env = RandomWalkBandit(n_arms=10, seed=0)
    agent = EGreedyAgent(epsilon=epsilon, step_size=step_size)
    agent.init(env)
    rewards, actions = util.run_episodes(env, agent, episodes=2000, steps=3000)
    mean_rewards.append(rewards)
    optimal_actions.append(actions)

util.plot_step_size_comparison(
    epsilon, step_sizes, mean_rewards, optimal_actions)
plt.show()

# <markdowncell>

# ### Optimistic Initial Values

# The action value estimation discussed so far depends on the initial estimate,
# $Q_{1}(a)$

# $$
#   Q_{k+1} = (1 - \alpha)^{k} Q_1 +
#             \sum_{i=1}^{k} \alpha ( 1 - \alpha) ^ {k-i} R_{i}
# $$

# Initial action values can be used as a simple way of encouraging exploration.
# Suppose that instead of setting the initial action values to zero, as we did
# in the 10-armed testbed, we set them all to +5. Recall that the $q^{*}(a)$
# in this problem are selected from a normal distribution with mean 0 and
# variance 1. An initial estimate of +5 is thus wildly optimistic.
# But this optimism encourages action-value methods to explore.
# Whichever actions are initially selected, the reward is less than
# the starting estimates; the learner switches to other actions, being
# "disappointed" with the rewards it is receiving.

# Let's run experiment, using `Bandit` class

# <codecell>

epsilons = [0, 0.1]
initial_qs = [5, 0]
mean_rewards, optimal_actions = [], []
for epsilon, initial_q in zip(epsilons, initial_qs):
    print('Running initial Q = {}...'.format(initial_q))
    env = Bandit(n_arms=10, seed=0)
    agent = EGreedyAgent(
        epsilon=epsilon, initial_q=initial_q, step_size=0.1)
    agent.init(env)
    rewards, actions = util.run_episodes(env, agent, episodes=2000, steps=1000)
    mean_rewards.append(rewards)
    optimal_actions.append(actions)

util.plot_initial_value_comparison(
    epsilons, initial_qs, mean_rewards, optimal_actions)
plt.show()
