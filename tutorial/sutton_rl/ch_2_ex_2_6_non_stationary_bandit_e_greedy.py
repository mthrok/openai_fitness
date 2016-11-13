"""Acswer script of excercise 2.6 from Reinforcement Learning by
 Richard S. Sutton and Andrew G. Barto

Demonstarates that simple averaging e-greedy agent does not work well
when the environment is not stattionary
"""
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

import luchador.env
import luchador.agent


class RandomWalkBandit(luchador.env.StationaryBandit):
    """Bandit problem with true action values taking independent random walks"""
    def __init__(self, n_arms, seed=None):
        super(RandomWalkBandit, self).__init__(n_arms, seed)

        # For separating the effect of random walk and random sampling
        self.rng_random_walk = np.random.RandomState(seed=0)

    def reset(self):
        outcome = super(RandomWalkBandit, self).reset()
        for dist in self.distributions:
            dist.mean = 5.0
        return outcome

    def step(self, n):
        outcome = super(RandomWalkBandit, self).step(n)
        for dist in self.distributions:
            dist.mean += self.rng_random_walk.randn()
        return outcome


def run_episode(env, agent, steps):
    agent.reset(env.reset().observation)

    rewards, optimal_actions = [], []
    for _ in range(steps):
        optimal_action = np.argmax([dist.mean for dist in env.distributions])
        action = agent.act()
        outcome = env.step(action)
        agent.observe(action, outcome)

        rewards.append(outcome.reward)
        optimal_actions.append(action == optimal_action)
    return rewards, optimal_actions


def run_episodes(env, agent, episodes, steps):
    mean_rewards, optimal_action_ratio = 0, 0
    for _ in range(episodes):
        rewards, opt_actions = run_episode(env, agent, steps)
        mean_rewards += np.asarray(rewards) / episodes
        optimal_action_ratio += np.asarray(opt_actions, dtype=float) / episodes
    return mean_rewards, optimal_action_ratio


def plot_result(epsilon, step_sizes, rewards, optimal_action_ratios):
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    for reward, ratio, step in zip(rewards, optimal_action_ratios, step_sizes):
        label = 'Step Size: {}, Epsilon: {:4.2f}'.format(step, epsilon)
        ax1.plot(reward, label=label)
        ax2.plot(100 * ratio, label=label)
    ax1.set_ylim(ymin=0)
    ax1.set_xlim(xmin=-10)
    ax1.legend(loc=4)
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Average Rewards')

    ax2.set_ylim(ymin=0, ymax=100)
    ax2.set_xlim(xmin=-10)
    ax2.legend(loc=4)
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Optimal Action Ratio [%]')


def main():
    epsilon = 0.1
    step_sizes = ['average', 0.1]
    mean_rewards, optimal_actions = [], []
    for step_size in step_sizes:
        env = RandomWalkBandit(n_arms=10, seed=0)
        agent = luchador.agent.EGreedyAgent(epsilon=epsilon, step_size=step_size)
        agent.init(env)
        rewards, actions = run_episodes(env, agent, episodes=2000, steps=3000)
        mean_rewards.append(rewards)
        optimal_actions.append(actions)

    plot_result(epsilon, step_sizes, mean_rewards, optimal_actions)
    plt.show()


if __name__ == '__main__':
    main()
