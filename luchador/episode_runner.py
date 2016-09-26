from __future__ import absolute_import

import logging

_LG = logging.getLogger(__name__)


__all__ = ['EpisodeRunner']


class EpisodeRunner(object):
    """Class for runnig episode"""
    def __init__(self, env, agent, max_timesteps=1000):
        self.env = env
        self.agent = agent
        self.max_timesteps = max_timesteps
        self.n_episodes = 0

    def _reset(self):
        """Reset environment and agent"""
        outcome = self.env.reset()
        self.agent.reset(outcome.observation)

    def _perform_post_episode_task(self):
        """Perform post episode task"""
        self.agent.perform_post_episode_task()

    def run_episode(self, max_timesteps=None):
        """Run one episode"""
        max_timesteps = max_timesteps or self.max_timesteps
        self._reset()

        total_rewards = 0
        for t in range(1, max_timesteps+1):
            action = self.agent.act()
            outcome = self.env.step(action)

            self.agent.observe(action, outcome)
            total_rewards += outcome.reward

            if outcome.terminal:
                break
        else:
            t = -1
        self._perform_post_episode_task()
        return t, total_rewards
