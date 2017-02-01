"""Module for building neural Q learning network"""
from __future__ import division
from __future__ import absolute_import

import logging

import luchador.util
from luchador import nn

_LG = logging.getLogger(__name__)

__all__ = ['DeepQLearning']


class DeepQLearning(luchador.util.StoreMixin, object):
    """Build Q-learning network and optimization operations

    Parameters
    ----------
    discout_rate : float
        Discount rate for computing future reward. Valid value range is
        (0.0, 1.0)

    scale_reward : number or None
        When given, reward is divided by this number before applying min/max
        threashold

    min_reward : number or None
        When given, clip reward after scaling.

    max_reward : number or None
        See `min_reward`.

    min_delta : number or None
        When given, error between predicted Q and target Q is clipped with
        this value.

    max_delta : number or None
        See `max_reward`
    """
    def __init__(
            self, discount_rate, scale_reward=None,
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
        self.vars = {
            'state0': None,
            'action': None,
            'reward': None,
            'state1': None,
            'terminal': None,
            'action_value_0': None,
            'target_q': None,
            'error': None,
        }
        self.models = {
            'pre_trans': None,
            'post_trans': None,
        }
        self.ops = {
            'sync': None,
        }

    def _validate_args(self, min_reward=None, max_reward=None,
                       min_delta=None, max_delta=None, **_):
        if (min_reward and not max_reward) or (max_reward and not min_reward):
            raise ValueError(
                'When clipping reward, both `min_reward` '
                'and `max_reward` must be provided.')
        if (min_delta and not max_delta) or (max_delta and not min_delta):
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

    def build(self, model_maker):
        """Build computation graph (error and sync ops) for Q learning

        Args:
          q_network(TFModel): TFModel which represetns Q network.
            Model must be pre-built since this function needs shape
            information.
        """
        with nn.variable_scope('pre_trans'):
            self.models['pre_trans'] = model_maker()
            self.vars['state0'] = self.models['pre_trans'].input
        with nn.variable_scope('post_trans'):
            self.models['post_trans'] = model_maker()
            self.vars['state1'] = self.models['post_trans'].input
        with nn.variable_scope('target_q_value'):
            self._build_target_q_value()
        with nn.variable_scope('sync'):
            self._build_sync_op()
        with nn.variable_scope('error'):
            self._build_error()
        return self

    ###########################################################################
    def _build_target_q_value(self):
        self.vars['terminal'] = terminal = nn.Input(
            shape=(None,), name='terminal')
        self.vars['reward'] = reward = nn.Input(
            shape=(None,), name='rewards')
        self.vars['action_value_0'] = self.models['pre_trans'].output

        if self.args['scale_reward']:
            reward = reward / self.args['scale_reward']
        if self.args['min_reward'] and self.args['max_reward']:
            min_val, max_val = self.args['min_reward'], self.args['max_reward']
            reward = reward.clip(min_value=min_val, max_value=max_val)

        max_q = self.models['post_trans'].output.max(axis=1)
        discounted = max_q * self.args['discount_rate']
        target_q = reward + (1.0 - terminal) * discounted

        n_actions = self.models['pre_trans'].output.shape[1]
        target_q = target_q.reshape([-1, 1]).tile([1, n_actions])
        self.vars['target_q'] = target_q

    ###########################################################################
    def _build_sync_op(self):
        src_vars = self.models['pre_trans'].get_parameter_variables()
        tgt_vars = self.models['post_trans'].get_parameter_variables()
        self.ops['sync'] = nn.build_sync_op(src_vars, tgt_vars, name='sync')

    def _build_error(self):
        self.vars['action'] = action = nn.Input(
            shape=(None,), dtype='uint8', name='action')

        min_, max_ = self.args['min_delta'], self.args['max_delta']
        sse2 = nn.cost.SSE2(min_delta=min_, max_delta=max_, elementwise=True)
        error = sse2(self.vars['target_q'], self.vars['action_value_0'])

        n_actions = self.models['pre_trans'].output.shape[1]
        mask = action.one_hot(n_classes=n_actions)

        self.vars['error'] = (mask * error).mean()
