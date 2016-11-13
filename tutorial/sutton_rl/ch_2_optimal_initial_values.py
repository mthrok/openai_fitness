"""Reproduction of Fig.2.1 in Chapter 2.2. Action-Value Methods from
Reinforcement Learning by Richard S. Sutton and Andrew G. Barto

Agents with different epsilon value for e-greedy policy are run against
statioinary n-arm bandit problem.

The result illustrates that greedy policy is not necessarily good for
longe term, which implies the importance of balancing exploitation and
exploration.
"""
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

import luchador.env
import luchador.agent


def run_episode(env, agent, steps):
    agent.reset(env.reset().observation)
    optimal_action = np.argmax([dist.mean for dist in env.distributions])

    rewards, optimal_actions = [], []
    for _ in range(steps):
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


def plot_result(epsilons, q_vals, rewards, optimal_action_ratios):
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    for reward, ratio, ep, q, in zip(
            rewards, optimal_action_ratios, epsilons, q_vals):
        label = 'Initial Q: {:4.2f}, Epsilon: {:4.2f}'.format(q, ep)
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
    epsilons = [0, 0.1]
    initial_qs = [5, 0]
    mean_rewards, optimal_actions = [], []
    for epsilon, initial_q in zip(epsilons, initial_qs):
        env = luchador.env.StationaryBandit(n_arms=10, seed=0)
        agent = luchador.agent.EGreedyAgent(
            epsilon=epsilon, initial_q=initial_q, step_size=0.1)
        agent.init(env)
        rewards, actions = run_episodes(env, agent, episodes=2000, steps=1000)
        mean_rewards.append(rewards)
        optimal_actions.append(actions)

    plot_result(epsilons, initial_qs, mean_rewards, optimal_actions)
    plt.show()


if __name__ == '__main__':
    main()
