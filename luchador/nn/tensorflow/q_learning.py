from __future__ import division
from __future__ import absolute_import

import logging

from luchador.nn.base.q_learning import BaseDeepQLearning
from . import scope, wrapper, cost, misc

__all__ = ['DeepQLearning']

_LG = logging.getLogger(__name__)


class DeepQLearning(BaseDeepQLearning):
    """Implement DeepQLearning

    See :any:`BaseDeepQLearning` for detail.
    """
    def build(self, model_maker):
        """Build computation graph (error and sync ops) for Q learning

        Args:
          q_network(TFModel): TFModel which represetns Q network.
            Model must be pre-built since this function needs shape
            information.
        """
        with scope.variable_scope('pre_trans'):
            self.pre_trans_net = model_maker()
            self.pre_states = self.pre_trans_net.input
        with scope.variable_scope('post_trans'):
            self.post_trans_net = model_maker()
            self.post_states = self.post_trans_net.input
        with scope.variable_scope('target_q_value'):
            self._build_target_q_value()
        with scope.variable_scope('sync'):
            self._build_sync_op()
        with scope.variable_scope('error'):
            self._build_error()
        return self

    ###########################################################################
    def _build_target_q_value(self):
        self.terminals = wrapper.Input(
            shape=(None,), name='terminals')
        self.rewards = wrapper.Input(
            shape=(None,), name='rewards')
        self.predicted_q = self.pre_trans_net.output

        rewards = self.rewards
        if self.args['scale_reward']:
            rewards = rewards / self.args['scale_reward']
        if self.args['min_reward'] and self.args['max_reward']:
            min_val, max_val = self.args['min_reward'], self.args['max_reward']
            rewards = rewards.clip(min_value=min_val, max_value=max_val)

        max_q = self.post_trans_net.output.max(axis=1)
        discounted = max_q * self.args['discount_rate']
        target_q = rewards + (1.0 - self.terminals) * discounted

        n_actions = self.pre_trans_net.output.shape[1]
        self.target_q = target_q.reshape([-1, 1]).tile([1, n_actions])

    ###########################################################################
    def _build_sync_op(self):
        src_vars = self.pre_trans_net.get_parameter_variables()
        tgt_vars = self.post_trans_net.get_parameter_variables()
        self.sync_op = misc.build_sync_op(src_vars, tgt_vars, name='sync')

    def _build_error(self):
        self.actions = wrapper.Input(
            shape=(None,), dtype='uint8', name='actions')

        min_, max_ = self.args['min_delta'], self.args['max_delta']
        sse2 = cost.SSE2(min_delta=min_, max_delta=max_, elementwise=True)
        error = sse2(self.target_q, self.predicted_q)

        n_actions = self.pre_trans_net.output.shape[1]
        mask = self.actions.one_hot(n_classes=n_actions)

        self.error = (mask * error).mean()
