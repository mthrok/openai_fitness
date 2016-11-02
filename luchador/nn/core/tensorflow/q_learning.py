from __future__ import division
from __future__ import absolute_import

import logging

import tensorflow as tf

from ..base import q_learning as base_q
from . import wrapper, cost

__all__ = ['DeepQLearning']

_LG = logging.getLogger(__name__)


class DeepQLearning(base_q.BaseDeepQLearning):
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
        with tf.variable_scope('sync'):
            self._build_sync_op()

        with tf.variable_scope('target_q_value'):
            self._build_target_q_value()

        with tf.variable_scope('error'):
            self._build_error()
        return self

    ###########################################################################
    def _build_target_q_value(self):
        with tf.name_scope('future_reward'):
            future = self._get_future_reward()

        with tf.name_scope('target_q_value'):
            target_q = self._get_target_q_value(future)

        self.future_reward = wrapper.Tensor(tensor=future)
        self.target_q = wrapper.Tensor(tensor=target_q)
        self.predicted_q = self.pre_trans_net.output

    def _get_future_q_value(self):
        self.discount_rate = tf.constant(
            self.args['discount_rate'], name='discount_rate')
        self.terminals = wrapper.Input(shape=(None,), name='terminals')
        terminals = self.terminals().unwrap()

        q = self.post_trans_net.output.unwrap()
        q = tf.reduce_max(q, reduction_indices=1)
        q = tf.mul(q, self.discount_rate)
        q = tf.mul(q, 1.0 - terminals)
        return q

    def _get_future_reward(self):
        with tf.name_scope('future_q_value'):
            post_q = self._get_future_q_value()

        self.rewards = wrapper.Input(shape=(None,), name='rewards')
        rewards = self.rewards().unwrap()
        if self.args['scale_reward']:
            scale_reward = tf.constant(
                self.args['scale_reward'], name='scale_reward')
            rewards = tf.truediv(rewards, scale_reward)
        if self.args['min_reward'] and self.args['max_reward']:
            min_reward = tf.constant(
                self.args['min_reward'], name='min_reward')
            max_reward = tf.constant(
                self.args['max_reward'], name='max_reward')
            rewards = tf.clip_by_value(rewards, min_reward, max_reward)

        future = tf.add(rewards, post_q, name='future_reward')
        return future

    def _get_target_q_value(self, future):
        n_actions = self.pre_trans_net.output.shape[1]

        self.actions = wrapper.Input(
            dtype='int32', shape=(None,), name='actions')
        actions = self.actions().unwrap()
        with tf.name_scope('reshape_current_q_value'):
            mask_off = tf.one_hot(actions, depth=n_actions, on_value=0.,
                                  off_value=1., name='actions_not_taken')
            current = self.pre_trans_net.output.unwrap()
            current = current * mask_off

        with tf.name_scope('reshape_future_q_value'):
            mask_on = tf.one_hot(actions, depth=n_actions, on_value=1.,
                                 off_value=0., name='actions_taken')
            future = tf.reshape(future, (-1, 1))
            future = tf.tile(future, tf.pack([1, n_actions]))
            future = future * mask_on

        target_q = tf.add(current, future, name='target_q')
        return target_q

    ###########################################################################
    def _build_sync_op(self):
        src_vars = self.pre_trans_net.get_parameter_variables()
        tgt_vars = self.post_trans_net.get_parameter_variables()
        ops = [tgt.unwrap().assign(src.unwrap())
               for src, tgt in zip(src_vars, tgt_vars)]
        self.sync_op = wrapper.Operation(op=tf.group(*ops, name='sync'))

    def _build_error(self):
        min_delta, max_delta = self.args['min_delta'], self.args['max_delta']
        sse2 = cost.SSE2(min_delta=min_delta, max_delta=max_delta)
        self.error = sse2(self.target_q, self.predicted_q)
