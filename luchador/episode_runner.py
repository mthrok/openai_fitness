"""Simple episode runner module"""
from __future__ import absolute_import

import time
import logging

_LG = logging.getLogger(__name__)


__all__ = ['EpisodeRunner']


class EpisodeRunner(object):
    """Class for runnig episode"""
    def __init__(self, env, agent, max_steps=10000):
        self.env = env
        self.agent = agent
        self.max_steps = max_steps

        self.episode = 0
        self.time = 0
        self.steps = 0
        self.rewards = 0

    def _reset(self):
        """Reset environment and agent"""
        self.episode += 1
        outcome = self.env.reset()
        self.agent.reset(outcome.state)
        return outcome

    def _perform_post_episode_task(self, stats):
        """Perform post episode task"""
        self.agent.perform_post_episode_task(stats)

    def run_episode(self, max_steps=None):
        """Run one episode"""
        steps, rewards = 0, 0
        t_start = time.time()

        outcome = self._reset()
        state0 = outcome.state
        steps += 1
        rewards += outcome.reward

        for _ in range(max_steps or self.max_steps):
            action = self.agent.act()

            outcome = self.env.step(action)
            state1 = outcome.state
            steps += 1
            rewards += outcome.reward

            self.agent.learn(
                state0, action, outcome.reward, state1,
                outcome.terminal, outcome.info)

            if outcome.terminal:
                break
            state0 = state1

        t_elapsed = time.time() - t_start

        self.time += t_elapsed
        self.steps += steps
        self.rewards += rewards

        stats = {
            'episode': self.episode,
            'rewards': rewards,
            'steps': steps,
            'time': t_elapsed,
        }
        self._perform_post_episode_task(stats)
        return stats
