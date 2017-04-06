"""Test Q-Learning module"""
from __future__ import division
from __future__ import absolute_import

import numpy as np

import luchador
import luchador.nn as nn
from luchador.nn import util
from luchador.agent.rl.q_learning import DeepQLearning
from tests.unit import fixture

_CONV = luchador.get_nn_conv_format()


def _make_dqn(
        discount_rate=0.9, min_reward=-1, max_reward=1,
        scale_reward=1.0, input_shape=None, n_actions=5, model_def=None):
    """Make DQN module for test

    If model_def is given, that model definition is used, otherwise,
    vanilla_dqn is used.
    """
    dqn = DeepQLearning(
        q_learning_config={
            'discount_rate': discount_rate,
            'min_reward': min_reward,
            'max_reward': max_reward,
            'scale_reward': scale_reward,
        },
        optimizer_config={
            'typename': 'RMSProp',
            'args': {
                'decay': 0.95,
                'epsilon': 1e-6,
                'learning_rate': 2.5e-4,
            }
        },
    )
    if model_def is None:
        if input_shape is None:
            input_shape = [
                None, 4, 84, 84] if _CONV == 'NCHW' else [None, 84, 84, 4]
        model_def = util.get_model_config(
            'vanilla_dqn', input_shape=input_shape, n_actions=n_actions,
        )
    session = nn.Session()
    dqn.build(model_def, session)
    session.initialize()
    return dqn


class DQNTest(fixture.TestCase):
    """Test DQN functions"""
    def test_sync(self):
        """Sync operation copy model0 network parameters to model1"""
        shape = [None, 4, 84, 84] if _CONV == 'NCHW' else [None, 84, 84, 4]
        model_def = util.get_model_config(
            'vanilla_dqn', input_shape=shape, n_actions=5)
        # Skip biases as they are deterministically initialized
        for cfg in model_def['args']['layer_configs']:
            if cfg['typename'] in ['Conv2D', 'Dense']:
                cfg['args']['with_bias'] = False

        with nn.variable_scope(self.get_scope()):
            dqn = _make_dqn(model_def=model_def)

        params0 = dqn.models['model_0'].get_parameters_to_train()
        params1 = dqn.models['model_1'].get_parameters_to_train()

        # check that variables are different before sync
        vars0 = dqn.session.run(outputs=params0)
        vars1 = dqn.session.run(outputs=params1)
        for var0, var1 in zip(vars0, vars1):
            with self.assertRaises(AssertionError):
                np.testing.assert_almost_equal(var0, var1)

        dqn.sync_network()

        # check that variables are equal after sync
        vars0 = dqn.session.run(outputs=params0)
        vars1 = dqn.session.run(outputs=params1)
        for var0, var1 in zip(vars0, vars1):
            np.testing.assert_almost_equal(var0, var1)

    def test_target_q(self):
        """Target Q value is correct"""
        discount, batch, n_actions = 0.9, 32, 5
        scale_reward, min_reward, max_reward = 3.0, -1, 1

        with nn.variable_scope(self.get_scope()):
            dqn = _make_dqn(
                discount_rate=discount,
                n_actions=n_actions, scale_reward=scale_reward)

        action_value_1 = np.random.randn(batch, n_actions)
        rewards = np.random.randn(batch,)
        terminal = np.random.randint(low=0, high=2, size=(batch,))

        target_q = dqn.session.run(
            outputs=dqn.vars['target_q'],
            inputs={
                dqn.vars['action_value_1']: action_value_1,
                dqn.vars['reward']: rewards,
                dqn.vars['terminal']: terminal,
            }
        )

        rewards = np.clip(rewards / scale_reward, min_reward, max_reward)
        max_action = np.max(action_value_1, axis=1)
        target_q_ = rewards + (1.0 - terminal) * discount * max_action
        target_q_ = np.tile(target_q_.reshape(-1, 1), (1, n_actions))

        np.testing.assert_almost_equal(target_q, target_q_, decimal=4)
