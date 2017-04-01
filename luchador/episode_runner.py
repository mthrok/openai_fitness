"""Simple episode runner module"""
from __future__ import absolute_import

import time
import logging

_LG = logging.getLogger(__name__)


__all__ = ['EpisodeRunner']


class _Counter(object):
    """Helper class for tracking count"""
    def __init__(self):
        self.episode = 0
        self.time = 0
        self.steps = 0
        self.rewards = 0

    def reset(self):
        """Increase episode count"""
        self.episode += 1

    def update(self, steps, rewards, time_):
        """Update total numbers"""
        self.time += time_
        self.steps += steps
        self.rewards += rewards


class EpisodeRunner(object):
    """Class for runnig episode"""
    def __init__(self, env, agent, max_steps=10000):
        self.env = env
        self.agent = agent
        self.max_steps = max_steps

        self._counter = _Counter()

    @property
    def episode(self):
        """Get the current episode number"""
        return self._counter.episode

    @property
    def steps(self):
        """Get the number of total steps taken"""
        return self._counter.steps

    @property
    def time(self):
        """Get the total time spent on running episodes in second"""
        return self._counter.time

    def _reset(self):
        """Reset environment and agent"""
        self._counter.reset()
        outcome = self.env.reset()
        self.agent.reset(outcome.state)
        return outcome

    def _perform_post_episode_task(self, stats):
        """Perform post episode task"""
        self._counter.update(
            rewards=stats['rewards'], steps=stats['steps'],
            time_=stats['time'])
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

        stats = {
            'episode': self.episode,
            'rewards': rewards,
            'steps': steps,
            'time': time.time() - t_start,
        }
        self._perform_post_episode_task(stats)
        return stats
