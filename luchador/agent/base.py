from __future__ import absolute_import


class Agent(object):
    def __init__(self, env, agent_config, global_config):
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def observe(self, action, observation, reward, done, info):
        """Observe the action and it's outcome.

        Args:
          action: The action that this agent previously took.
          observation: Observation (of environment) caused by the action.
          reward: Reward acquired by the action.
          done (bool): Indicates if a task is complete or not.
          info (dict): Infomation related to environment.

          observation, reward, done, info are variables returned by
          environment. See gym.core:Env.step.
        """
        raise NotImplementedError('observe method is not implemented.')

    def act(self):
        """Choose action. Must be implemented in subclass."""
        raise NotImplementedError('act method is not implemented.')

    def reset(self, observation):
        """Reset agent with the initial state of the environment."""
        raise NotImplementedError('reset method is not implemented.')

    def perform_post_episode_task(self):
        """Perform post episode task"""
        pass
