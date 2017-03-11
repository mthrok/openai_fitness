"""Module for building neural Q learning network"""
from __future__ import division
from __future__ import absolute_import

import logging
from collections import OrderedDict

import numpy as np

import luchador.util
from luchador import nn

_LG = logging.getLogger(__name__)

__all__ = ['DeepQLearning', 'DoubleDeepQLearning']


def _validate_q_learning_config(min_reward=None, max_reward=None, **_):
    if (min_reward and not max_reward) or (max_reward and not min_reward):
        raise ValueError(
            'When clipping reward, both `min_reward` '
            'and `max_reward` must be provided.')


def _make_model(model_def, scope):
    with nn.variable_scope(scope):
        model = nn.make_model(model_def)
        state = model.input
        action_value = model.output
    return model, state, action_value


def _build_sync_op(src_model, tgt_model, scope):
    with nn.variable_scope(scope):
        src_vars = src_model.get_parameters_to_serialize()
        tgt_vars = tgt_model.get_parameters_to_serialize()
        return nn.ops.build_sync_op(src_vars, tgt_vars, name='sync')


def _build_error(target_q, action_value_0, action):
    n_actions = action_value_0.shape[1]
    delta = (target_q - action_value_0)
    error = nn.ops.minimum(nn.ops.abs(delta), (delta * delta))
    mask = nn.ops.one_hot(action, n_classes=n_actions, dtype=error.dtype)
    return nn.ops.reduce_sum(mask * error, axis=1)


def _clip_grads(grads_and_vars, clip_norm):
    """Clip gradients by norm

    Parameters
    ----------
    grads_and_vars : list
        Gradient and Variable tuples. Return value from ``compute_gradients``.

    clip_norm : Number or Tensor
        Value to clip gradients

    Returns
    -------
    list
        Resulting gradients and vars pairs
    """
    ret = []
    for grad, var in grads_and_vars:
        name = '{}_clip'.format(grad.name)
        grad = nn.ops.clip_by_norm(grad, clip_norm=clip_norm, name=name)
        ret.append((grad, var))
    return ret


class DeepQLearning(luchador.util.StoreMixin, object):
    """Implement Neural Network part of DQN [1]_:

    Parameters
    ----------
    q_learning_config : dict
        Configuration for building target Q value.

        discout_rate : float
            Discount rate for computing future reward. Valid value range is
            (0.0, 1.0)
        scale_reward : number or None
            When given, reward is divided by this number before applying
            min/max threashold
        min_reward : number or None
            When given, clip reward after scaling.
        max_reward : number or None
            See `min_reward`.

    optimizer_config : dict
        Configuration for optimizer

        name: str
            The name of cost class. See :py:mod:`luchador.nn.base.optimizer`
            for the list of available classes.
        args : dict
            Configuration for the optimizer class

    References
    ----------
    .. [1] Mnih, V et. al (2015)
        Human-level control through deep reinforcement learning
        https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
    """
    # pylint: disable=too-many-instance-attributes
    def __init__(self, q_learning_config, optimizer_config, clip_grads=None):
        self._store_args(
            q_learning_config=q_learning_config,
            optimizer_config=optimizer_config,
            clip_grads=clip_grads,
        )
        self.vars = None
        self.models = None
        self.ops = None
        self.optimizer = None
        self.session = None

    def _validate_args(self, q_learning_config=None, clip_grads=None, **_):
        if q_learning_config is not None:
            _validate_q_learning_config(**q_learning_config)
        if clip_grads:
            if 'clip_norm' not in clip_grads:
                raise ValueError('`clip_norm` must be given in `clip_grads`')

    def build(self, model_def, initial_parameter):
        """Build computation graph (error and sync ops) for Q learning

        Parameters
        ----------
        n_actions: int
            The number of available actions in the environment.
        """
        # pylint: disable=too-many-locals
        model_0, state_0, action_value_0 = _make_model(model_def, 'pre_trans')
        model_1, state_1, action_value_1 = _make_model(model_def, 'post_trans')
        sync_op = _build_sync_op(model_0, model_1, 'sync')

        with nn.variable_scope('target_q_value'):
            reward = nn.Input(shape=(None,), name='rewards')
            terminal = nn.Input(shape=(None,), name='terminal')
            target_q, post_q = self._build_target_q_value(
                action_value_1, reward, terminal)

        with nn.variable_scope('error'):
            action_0 = nn.Input(
                shape=(None,), dtype='int32', name='action_0')
            error = _build_error(target_q, action_value_0, action_0)

        weight = nn.Input(
            shape=(None,), name='sample_weight')
        self._init_optimizer()
        optimize_op = self._build_optimize_op(
            loss=nn.ops.reduce_mean(error * weight),
            params=model_0.get_parameters_to_train())

        self._init_session(initial_parameter)

        self.models = {
            'model_0': model_0,
            'model_1': model_1,
        }
        self.vars = {
            'state_0': state_0,
            'state_1': state_1,
            'action_value_0': action_value_0,
            'action_value_1': action_value_1,
            'action_0': action_0,
            'reward': reward,
            'terminal': terminal,
            'post_q': post_q,
            'target_q': target_q,
            'error': error,
            'weight': weight,
        }
        self.ops = {
            'sync': sync_op,
            'optimize': optimize_op,
        }

    def _build_target_q_value(self, action_value_1, reward, terminal):
        config = self.args['q_learning_config']
        # Clip rewrads
        if 'scale_reward' in config:
            reward = reward / config['scale_reward']
        if 'min_reward' in config and 'max_reward' in config:
            min_val, max_val = config['min_reward'], config['max_reward']
            reward = nn.ops.clip_by_value(
                reward, min_value=min_val, max_value=max_val)

        # Build Target Q
        post_q = nn.ops.reduce_max(action_value_1, axis=1)
        discounted_q = post_q * config['discount_rate']
        target_q = reward + (1.0 - terminal) * discounted_q

        n_actions = action_value_1.shape[1]
        target_q = nn.ops.tile(
            nn.ops.reshape(target_q, [-1, 1]), [1, n_actions])
        return target_q, post_q

    def _build_optimize_op(self, loss, params):
        grads_and_vars = nn.ops.compute_gradient(loss=loss, wrt=params)
        # Remove untrainable variables
        grads_and_vars = [g_v for g_v in grads_and_vars if g_v[0] is not None]
        if self.args.get('clip_grads'):
            grads_and_vars = _clip_grads(
                grads_and_vars, **self.args['clip_grads'])
        return self.optimizer.apply_gradients(grads_and_vars)

    ###########################################################################
    def _init_optimizer(self):
        cfg = self.args['optimizer_config']
        self.optimizer = nn.get_optimizer(cfg['typename'])(**cfg['args'])

    def _init_session(self, initial_parameter=None):
        self.session = nn.Session()
        if initial_parameter:
            _LG.info('Loading parameters from %s', initial_parameter)
            self.session.load_from_file(initial_parameter)
        else:
            self.session.initialize()

    ###########################################################################
    def predict_action_value(self, state):
        """Predict action values

        Parameters
        ----------
        state : NumPy ND Array
            Environment state

        Returns
        -------
        NumPy ND Array
            Action values
        """
        return self.session.run(
            outputs=self.vars['action_value_0'],
            inputs={self.vars['state_0']: state},
            name='action_value0',
        )

    def sync_network(self):
        """Synchronize parameters of model_1 with those of model_0"""
        self.session.run(updates=self.ops['sync'], name='sync')

    def train(self, state0, action, reward, state1, terminal, weight=None):
        """Train model network

        Parameters
        ----------
        state0 : NumPy ND Array
            Environment states before taking actions

        action : NumPy ND Array
            Actions taken in state0

        reward : NumPy ND Array
            Rewards obtained by taking the action_0.

        state1 : NumPy ND Array
            Environment states after action_0 are taken

        terminal : NumPy ND Array
            Flags for marking corresponding states in state1 are
            terminal states.

        weight : NumPy ND Array
            Weight of each data points. Scale parameter update.

        Returns
        -------
        NumPy ND Array
            Mean error between Q prediction and target Q
        """
        if weight is None:
            weight = np.ones((action.size, ), dtype=self.vars['weight'].dtype)

        return self._train(state0, action, reward, state1, terminal, weight)

    def _train(self, state_0, action_0, reward, state_1, terminal, weight):
        updates = self.models['model_0'].get_update_operations()
        updates += [self.ops['optimize']]
        return self.session.run(
            outputs=self.vars['error'],
            inputs={
                self.vars['state_0']: state_0,
                self.vars['action_0']: action_0,
                self.vars['reward']: reward,
                self.vars['state_1']: state_1,
                self.vars['terminal']: terminal,
                self.vars['weight']: weight,
            },
            updates=updates,
            name='minibatch_training',
        )

    ###########################################################################
    def get_parameters_to_serialize(self):
        """Fetch network parameters and optimizer parameters for saving"""
        params = (
            self.models['model_0'].get_parameters_to_serialize() +
            self.optimizer.get_parameters_to_serialize()
        )
        params_val = self.session.run(outputs=params, name='save_params')
        return OrderedDict([
            (var.name, val) for var, val in zip(params, params_val)
        ])

    ###########################################################################
    def get_parameters_to_summarize(self):
        """Fetch parameters of each layer"""
        params = self.models['model_0'].get_parameters_to_serialize()
        params_vals = self.session.run(outputs=params, name='model_0_params')
        return {
            '/'.join(v.name.split('/')[1:]): val
            for v, val in zip(params, params_vals)
        }

    def get_layer_outputs(self, state):
        """Fetch outputs from each layer

        Parameters
        ----------
        state : NumPy ND Array
            Input to model_0 (pre-transition model)
        """
        outputs = self.models['model_0'].get_output_tensors()
        output_vals = self.session.run(
            outputs=outputs,
            inputs={self.vars['state_0']: state},
            name='model_0_outputs'
        )
        return {
            '/'.join(v.name.split('/')[1:]): val
            for v, val in zip(outputs, output_vals)
        }


class DoubleDeepQLearning(DeepQLearning):
    """Implement Neural Network part of Double DQN [1]_:

    References
    ----------
    .. [1] Hasselt, H et. al (2015)
        Deep Reinforcement Learning with Double Q-learning
        https://arxiv.org/abs/1509.06461
    """
    def _train(self, state_0, action_0, reward, state_1, terminal, weight):
        # Find the best action after state_1 by feeding state_1 to model_0
        action_value_1_0, action_value_1 = self.session.run(
            outputs=[
                self.vars['action_value_0'],
                self.vars['action_value_1'],
            ],
            inputs={
                self.vars['state_0']: state_1,
                self.vars['state_1']: state_1,
            },
            name='fetch_action',
        )
        post_q = action_value_1[
            [i for i in range(action_value_1.shape[0])],
            np.argmax(action_value_1_0, axis=1)
        ]
        updates = self.models['model_0'].get_update_operations()
        updates += [self.ops['optimize']]
        return self.session.run(
            outputs=self.vars['error'],
            inputs={
                self.vars['state_0']: state_0,
                self.vars['action_0']: action_0,
                self.vars['post_q']: post_q,
                self.vars['reward']: reward,
                self.vars['terminal']: terminal,
                self.vars['weight']: weight,
            },
            updates=updates,
            name='minibatch_training',
        )
