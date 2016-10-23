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
        outcome = self.env.reset()
        self.agent.reset(outcome.observation)
        self.episode += 1

    def _perform_post_episode_task(self, stats):
        """Perform post episode task"""
        self.agent.perform_post_episode_task(stats)

    def run_episode(self, max_steps=None):
        """Run one episode"""
        max_steps = max_steps or self.max_steps
        self._reset()

        steps, rewards = 0, 0
        t_start = time.time()
        for steps in range(1, max_steps+1):
            action = self.agent.act()
            outcome = self.env.step(action)

            self.agent.observe(action, outcome)
            rewards += outcome.reward

            if outcome.terminal:
                break

        t_elapsed = time.time() - t_start

        stats = {
            'episode': self.episode,
            'rewards': rewards,
            'steps': steps,
            'time': t_elapsed,
        }
        self.time += t_elapsed
        self.steps += steps
        self.rewards += rewards

        self._perform_post_episode_task(stats)
        return stats
