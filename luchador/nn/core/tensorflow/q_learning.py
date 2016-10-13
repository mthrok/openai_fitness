from __future__ import division
from __future__ import absolute_import

import logging

import tensorflow as tf

from luchador import get_nn_dtype
from ..base import DeepQLearning as BaseQLI
from .wrapper import Tensor, Input, Operation
from .cost import SSE2

__all__ = ['DeepQLearning']

_LG = logging.getLogger(__name__)


class DeepQLearning(BaseQLI):
    def build(self, model_maker):
        """Build computation graph (error and sync ops) for Q learning

        Args:
          q_network(TFModel): TFModel which represetns Q network.
            Model must be pre-built since this function needs shape
            information.
        """
        with tf.variable_scope('pre_trans'):
            self.pre_trans_net = model_maker()
            self.pre_states = self.pre_trans_net.input
        with tf.variable_scope('post_trans'):
            self.post_trans_net = model_maker()
            self.post_states = self.post_trans_net.input
        with tf.variable_scope('target_q_value'):
            self._build_target_q_value()
        with tf.variable_scope('sync'):
            self._build_sync_op()
        with tf.variable_scope('error'):
            self._build_error()
        return self

    def _build_target_q_value(self):
        dtype = get_nn_dtype()
        min_reward = self.args['min_reward']
        max_reward = self.args['max_reward']
        scale_reward = self.args['scale_reward']
        discount_rate = self.args['discount_rate']

        self.actions = Input(dtype='int32', shape=(None,), name='actions')
        self.rewards = Input(dtype=dtype, shape=(None,), name='rewards')
        self.terminals = Input(dtype=dtype, shape=(None,), name='terminals')

        actions = self.actions().unwrap()
        rewards = self.rewards().unwrap()
        terminals = self.terminals().unwrap()
        with tf.name_scope('future_reward'):
            with tf.name_scope('future_q_value'):
                post_q = self.post_trans_net.output.unwrap()
                post_q = tf.reduce_max(
                    post_q, reduction_indices=1, name='max_post_q')
                post_q = tf.mul(
                    post_q, discount_rate, name='discounted_max_post_q')
                post_q = tf.mul(
                    post_q, 1.0 - terminals, name='masked_max_post_q')

            if scale_reward:
                rewards = tf.truediv(
                    rewards, scale_reward, name='scaled_reward')
            if min_reward and max_reward:
                with tf.name_scope('clipped_reward'):
                    rewards = tf.clip_by_value(rewards, min_reward, max_reward)

            future = tf.add(rewards, post_q, name='future_reward')

        with tf.name_scope('target_q_value'):
            n_actions = self.pre_trans_net.output.get_shape()[1]
            with tf.name_scope('reshape_current_q_value'):
                mask_off = tf.one_hot(actions, depth=n_actions, on_value=0.,
                                      off_value=1., name='actions_not_taken')
                current = tf.identity(self.pre_trans_net.output.unwrap())
                current = current * mask_off

            with tf.name_scope('reshape_future_q_value'):
                mask_on = tf.one_hot(actions, depth=n_actions, on_value=1.,
                                     off_value=0., name='actions_taken')
                future = tf.reshape(future, (-1, 1))
                future = tf.tile(future, tf.pack([1, n_actions]))
                future = future * mask_on

            target_q = tf.add(current, future, name='target_q')

        self.future_reward = Tensor(tensor=future)
        self.target_q = Tensor(tensor=target_q)
        self.predicted_q = self.pre_trans_net.output

    def _build_sync_op(self):
        src_vars = self.pre_trans_net.get_parameter_variables()
        tgt_vars = self.post_trans_net.get_parameter_variables()
        ops = [tgt.unwrap().assign(src.unwrap())
               for src, tgt in zip(src_vars, tgt_vars)]
        self.sync_op = Operation(op=tf.group(*ops, name='sync'))

    def _build_error(self):
        min_delta, max_delta = self.args['min_delta'], self.args['max_delta']
        sse2 = SSE2(min_delta=min_delta, max_delta=max_delta)
        self.error = sse2(self.target_q, self.predicted_q)
