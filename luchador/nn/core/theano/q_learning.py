from __future__ import division
from __future__ import absolute_import

import logging
from collections import OrderedDict

import theano.tensor as T

from ..base import DeepQLearning as BaseQLI
from . import scope as scp
from .wrapper import Input, Tensor, Operation
from .cost import SSE2

_LG = logging.getLogger(__name__)

__all__ = ['DeepQLearning']


class DeepQLearning(BaseQLI):
    def build(self, model_maker):
        """Build computation graph (error and sync ops) for Q learning

        Args:
          q_network(TFModel): TFModel which represetns Q network.
            Model must be pre-built since this function needs shape
            information.
        """
        with scp.variable_scope('pre_trans'):
            self.pre_trans_net = model_maker()
            self.pre_states = self.pre_trans_net.input
        with scp.variable_scope('post_trans'):
            self.post_trans_net = model_maker()
            self.post_states = self.post_trans_net.input
        with scp.variable_scope('target_q_value'):
            self._build_target_q_value()
        with scp.variable_scope('sync'):
            self._build_sync_op()
        with scp.variable_scope('error'):
            self._build_error()
        return self

    def _build_target_q_value(self):
        min_reward = self.args['min_reward']
        max_reward = self.args['max_reward']
        discount_rate = self.args['discount_rate']

        self.actions = Input(dtype='uint16', shape=(None,), name='actions')
        self.rewards = Input(dtype='float64', shape=(None,), name='rewards')
        self.terminals = Input(shape=(None,), name='continuations')

        actions = self.actions().get()
        rewards = self.rewards().get()
        terminals = self.terminals().get()

        post_q = self.post_trans_net.output.get()
        post_q = T.max(post_q, axis=1)
        post_q = post_q * discount_rate
        post_q = post_q * (1.0 - terminals)

        if min_reward and max_reward:
            rewards = rewards.clip(min_reward, max_reward)
        future = rewards + post_q

        n_actions = self.pre_trans_net.output.get_shape()[1]
        mask_on = T.extra_ops.to_one_hot(actions, n_actions)
        mask_off = 1.0 - mask_on
        current = self.pre_trans_net.output.get()
        current = current * mask_off

        future = T.reshape(future, (-1, 1))
        future = T.tile(future, [1, n_actions])
        future = future * mask_on

        target_q = current + future
        # TODO: Add shape inference
        self.future_reward = Tensor(tensor=future, shape=(-1, ))
        self.target_q = Tensor(tensor=target_q, shape=(-1, n_actions))
        self.predicted_q = self.pre_trans_net.output

    def _build_sync_op(self):
        sync_op = OrderedDict()
        src_vars = self.pre_trans_net.get_parameter_variables()
        tgt_vars = self.post_trans_net.get_parameter_variables()
        for src, tgt in zip(src_vars, tgt_vars):
            sync_op[tgt.get()] = src.get()
        self.sync_op = Operation(op=sync_op)

    def _build_error(self):
        min_delta, max_delta = self.args['min_delta'], self.args['max_delta']
        sse2 = SSE2(min_delta=min_delta, max_delta=max_delta)
        self.error = sse2(self.target_q, self.predicted_q)
