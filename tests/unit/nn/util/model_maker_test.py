"""Test nn.util module"""
from __future__ import absolute_import

import numpy as np

from luchador import nn
from tests.unit import fixture

# pylint: disable=invalid-name


class ModelMakerTest(fixture.TestCase):
    """Test make_model functions"""
    def test_make_layer_with_reuse(self):
        """make_layer sets parameter variables correctly"""
        shape, scope, name = (3, 4), self.get_scope(), 'Dense'
        layer_config = {
            'typename': 'Dense',
            'args': {
                'n_nodes': 5,
                'name': 'Dense',
            },
            'parameters': {
                'weight': {
                    'typename': 'Variable',
                    'name': '{}/{}/weight'.format(scope, name)
                },
                'bias': {
                    'typename': 'Variable',
                    'name': '{}/{}/bias'.format(scope, name)
                },
            }
        }

        with nn.variable_scope(scope):
            layer1 = nn.layer.Dense(n_nodes=5, name=name)
            tensor = nn.Input(shape=shape)
            out1 = layer1(tensor)

        layer2 = nn.make_layer(layer_config)
        out2 = layer2(tensor)

        for key in ['weight', 'bias']:
            var1 = layer1.get_parameter_variables(key)
            var2 = layer2.get_parameter_variables(key)
            self.assertIs(var1, var2)

        session = nn.Session()
        session.initialize()

        input_val = np.random.rand(*shape)
        out1, out2 = session.run(
            outputs=[out1, out2],
            inputs={tensor: input_val}
        )

        np.testing.assert_almost_equal(
            out1, out2
        )

    def test_make_layer_with_reuse_in_scope(self):
        """make_layer sets parameter variables correctly"""
        shape, scope1, scope2, name = (3, 4), self.get_scope(), 'foo', 'dense'
        layer_config = {
            'typename': 'Dense',
            'args': {
                'n_nodes': 5,
                'name': name,
            },
            'parameters': {
                'weight': {
                    'typename': 'Variable',
                    'name': '{}/{}/weight'.format(scope2, name),
                },
                'bias': {
                    'typename': 'Variable',
                    'name': '{}/{}/bias'.format(scope2, name),
                },
            }
        }
        with nn.variable_scope(scope1):
            with nn.variable_scope(scope2):
                layer1 = nn.layer.Dense(n_nodes=5, name=name)
                tensor = nn.Input(shape=shape)
                out1 = layer1(tensor)

            layer2 = nn.make_layer(layer_config)
            out2 = layer2(tensor)

        for key in ['weight', 'bias']:
            var1 = layer1.get_parameter_variables(key)
            var2 = layer2.get_parameter_variables(key)
            self.assertIs(var1, var2)

        session = nn.Session()
        session.initialize()

        input_val = np.random.rand(*shape)
        out1, out2 = session.run(
            outputs=[out1, out2],
            inputs={tensor: input_val}
        )

        np.testing.assert_almost_equal(
            out1, out2
        )
