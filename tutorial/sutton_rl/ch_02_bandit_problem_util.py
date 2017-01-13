"""Reproduction of Fig.2.1 in Chapter 2.2. Action-Value Methods from
Reinforcement Learning by Richard S. Sutton and Andrew G. Barto

Agents with different epsilon value for e-greedy policy are run against
statioinary n-arm bandit problem.

The result illustrates that greedy policy is not necessarily good for
longe term, which implies the importance of balancing exploitation and
exploration.
"""
from __future__ import division
from __future__ import absolute_import

import numpy as np
import matplotlib.pyplot as plt


def _run_episode(env, agent, steps):
    agent.reset(env.reset().observation)

    rewards, optimal_actions = [], []
    for _ in range(steps):
        optimal_action = np.argmax(env.mean)
        action = agent.act()
        outcome = env.step(action)
        agent.observe(action, outcome)

        rewards.append(outcome.reward)
        optimal_actions.append(action == optimal_action)
    return rewards, optimal_actions


def run_episodes(env, agent, episodes, steps):
    mean_rewards, optimal_action_ratio = 0, 0
    for _ in range(episodes):
        rewards, opt_actions = _run_episode(env, agent, steps)
        mean_rewards += np.asarray(rewards) / episodes
        optimal_action_ratio += np.asarray(opt_actions, dtype=float) / episodes
    return mean_rewards, optimal_action_ratio


def plot_epsilon_comparison(epsilons, rewards, optimal_action_ratios):
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    for reward, ratio, eps in zip(rewards, optimal_action_ratios, epsilons):
        label = 'Epsilon: {:4.2f}'.format(eps)
        ax1.plot(reward, label=label)
        ax2.plot(100 * ratio, label=label)
    ax1.set_ylim(ymin=0)
    ax1.set_xlim(xmin=-10)
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Average Rewards')

    ax2.set_ylim(ymin=0, ymax=100)
    ax2.set_xlim(xmin=-10)
    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Optimal Action Ratio [%]')


def plot_step_size_comparison(
        epsilon, step_sizes, rewards, optimal_action_ratios):
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    for reward, ratio, step in zip(rewards, optimal_action_ratios, step_sizes):
        label = 'Epsilon: {:4.2f}, Step Size: {}'.format(epsilon, step)
        ax1.plot(reward, label=label)
        ax2.plot(100 * ratio, label=label)
    ax1.set_ylim(ymin=0)
    ax1.set_xlim(xmin=-10)
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Average Rewards')

    ax2.set_ylim(ymin=0, ymax=100)
    ax2.set_xlim(xmin=-10)
    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Optimal Action Ratio [%]')


def plot_initial_value_comparison(
        epsilons, q_vals, rewards, optimal_action_ratios):
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
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Average Rewards')

    ax2.set_ylim(ymin=0, ymax=100)
    ax2.set_xlim(xmin=-10)
    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Optimal Action Ratio [%]')
