from __future__ import division
from __future__ import absolute_import

import logging
from collections import OrderedDict

import theano.tensor as T

from luchador.nn.base.q_learning import BaseDeepQLearning
from . import (
    scope,
    wrapper,
    cost,
)

_LG = logging.getLogger(__name__)

__all__ = ['DeepQLearning']


class DeepQLearning(BaseDeepQLearning):
    """Implement DeepQLearning in Theano.

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
        future = self._get_future_reward()
        target_q = self._get_target_q_value(future)

        # TODO: Add shape inference
        n_actions = self.pre_trans_net.output.shape[1]
        self.future_reward = wrapper.Tensor(tensor=future, shape=(-1, ))
        self.target_q = wrapper.Tensor(tensor=target_q, shape=(-1, n_actions))
        self.predicted_q = self.pre_trans_net.output

    def _build_future_q_value(self):
        self.discount_rate = T.constant(self.args['discount_rate'])
        self.terminals = wrapper.Input(shape=(None,), name='terminals')
        terminals = self.terminals().unwrap()

        q_value = self.post_trans_net.output.unwrap()
        q_value = T.max(q_value, axis=1)
        q_value = q_value * self.discount_rate
        q_value = q_value * (1.0 - terminals)
        return q_value

    def _get_future_reward(self):
        post_q = self._build_future_q_value()

        self.rewards = wrapper.Input(
            dtype='float64', shape=(None,), name='rewards')
        rewards = self.rewards().unwrap()
        if self.args['scale_reward']:
            scale_reward = T.constant(self.args['scale_reward'])
            rewards = rewards / scale_reward
        if self.args['min_reward'] and self.args['max_reward']:
            min_reward = T.constant(self.args['min_reward'])
            max_reward = T.constant(self.args['max_reward'])
            rewards = rewards.clip(min_reward, max_reward)

        future = rewards + post_q
        return future

    def _get_target_q_value(self, future):
        n_actions = self.pre_trans_net.output.shape[1]

        self.actions = wrapper.Input(
            dtype='uint16', shape=(None,), name='actions')

        actions = self.actions().unwrap()
        mask_on = T.extra_ops.to_one_hot(actions, n_actions)
        mask_off = 1.0 - mask_on
        current = self.pre_trans_net.output.unwrap()
        current = current * mask_off

        future = T.reshape(future, (-1, 1))
        future = T.tile(future, [1, n_actions])
        future = future * mask_on

        target_q = current + future
        return target_q

    ###########################################################################
    def _build_sync_op(self):
        sync_op = OrderedDict()
        src_vars = self.pre_trans_net.get_parameter_variables()
        tgt_vars = self.post_trans_net.get_parameter_variables()
        for src, tgt in zip(src_vars, tgt_vars):
            sync_op[tgt.unwrap()] = src.unwrap()
        self.sync_op = wrapper.Operation(op=sync_op)

    def _build_error(self):
        min_delta, max_delta = self.args['min_delta'], self.args['max_delta']
        sse2 = cost.SSE2(min_delta=min_delta, max_delta=max_delta)
        self.error = sse2(self.target_q, self.predicted_q)
