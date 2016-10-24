from __future__ import absolute_import

import logging

from luchador.common import StoreMixin

__all__ = ['DeepQLearning']

_LG = logging.getLogger(__name__)


class DeepQLearning(StoreMixin, object):
    """Build Q-learning network and optimization operations

    Args:
      discout_rate (float): Discount rate for computing future reward.
                            Valid value range is (0.0, 1.0)

      scale_reward (number or None): When given, reward is divided by this
                                     number before applying min/max threashold
      min_reward(number or None): When given, clip reward after scaling.
      max_reward(number or None): See `min_reward`.

      min_delta(number or None): When given, error between predicted Q and
                                 target Q is clipped with this value.
      max_delta(number or None): See `max_reward`
    """
    def __init__(self, discount_rate, scale_reward=None,
                 min_reward=None, max_reward=None,
                 min_delta=None, max_delta=None):
        self._store_args(
            discount_rate=discount_rate,
            scale_reward=scale_reward,
            min_reward=min_reward,
            max_reward=max_reward,
            min_delta=min_delta,
            max_delta=max_delta,
        )

        # Inputs to the network
        self.pre_states = None
        self.actions = None
        self.rewards = None
        self.post_states = None
        self.terminals = None

        # Actual NN models
        self.pre_trans_net = None
        self.post_trans_net = None

        # Q values
        self.future_reward = None
        self.predicted_q = None
        self.target_q = None
        self.error = None

        # Sync operation
        self.sync_op = None

    def _validate_args(self, args):
        if (
                (args['min_reward'] or args['max_reward']) and
                (not args['max_reward'] or not args['min_reward'])
        ):
            raise ValueError(
                'When clipping reward, both `min_reward` '
                'and `max_reward` must be provided.')
        if (
                (args['min_delta'] or args['max_delta']) and
                (not args['max_delta'] or not args['min_delta'])
        ):
            raise ValueError(
                'When clipping reward, both `min_delta` '
                'and `max_delta` must be provided.')

    def __call__(self, q_network_maker):
        """Build computation graph (error and sync ops) for Q learning

        Args:
          q_network_maker(function): Model factory function which are called
            without any arguments and return Model object
        """
        self.build(q_network_maker)

    def build(self, q_network):
        """Build computation graph (error and sync ops) for Q learning"""
        raise NotImplementedError(
            '`build` method is not implemented for {}.{}.'
            .format(type(self).__module__, type(self).__name__)
        )
