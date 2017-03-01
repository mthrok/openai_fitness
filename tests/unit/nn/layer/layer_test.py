"""Test behaviors common to multiple layers"""
from __future__ import division
from __future__ import absolute_import

import numpy as np

# import theano
# theano.config.optimizer = 'None'
# theano.config.exception_verbosity = 'high'

from luchador import nn
from tests.unit.fixture import TestCase

# pylint: disable=invalid-name,too-many-locals,protected-access


class TestInitializer(TestCase):
    """Test initializer selection"""
    def test_dynamic_initializer(self):
        """Initializers are correctly selected"""
        n_in, n_nodes, weight_val, bias_val = 4, 5, 13, 7
        with nn.variable_scope(self.get_scope()):
            dense = nn.layer.Dense(
                n_nodes=5,
                initializers={
                    'weight': {
                        'typename': 'ConstantInitializer',
                        'args': {
                            'value': weight_val,
                        },
                    },
                    'bias': {
                        'typename': 'ConstantInitializer',
                        'args': {
                            'value': bias_val,
                        }
                    }
                }
            )
            dense(nn.Input(shape=(3, n_in)))

        session = nn.Session()
        session.initialize()

        weight, bias = session.run(
            outputs=[
                dense.get_parameter_variable('weight'),
                dense.get_parameter_variable('bias'),
            ]
        )

        np.testing.assert_almost_equal(
            weight, weight_val * np.ones((n_in, n_nodes)))
        np.testing.assert_almost_equal(
            bias, bias_val * np.ones((n_nodes,)))


class TestReuse(TestCase):
    """Test parameter reuse"""
    def test_paramter_reuse_dense(self):
        """Dense layer is built using existing Variables"""
        shape = (3, 5)
        with nn.variable_scope(self.get_scope()):
            layer1 = nn.layer.Dense(n_nodes=5)
            layer2 = nn.layer.Dense(n_nodes=5)

            tensor = nn.Input(shape=shape)
            out1 = layer1(tensor)
            layer2.set_parameter_variables(
                weight=layer1.get_parameter_variable('weight'),
                bias=layer1.get_parameter_variable('bias'),
            )
            out2 = layer2(tensor)

        for key in ['weight', 'bias']:
            var1 = layer1.get_parameter_variable(key)
            var2 = layer2.get_parameter_variable(key)
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

    def test_paramter_reuse_conv2d(self):
        """Conv2D layer is built using existing Variables"""
        shape = (10, 11, 12, 13)
        with nn.variable_scope(self.get_scope()):
            layer1 = nn.layer.Conv2D(
                filter_width=5, filter_height=3, n_filters=4, strides=1,
                padding='VALID')
            layer2 = nn.layer.Conv2D(
                filter_width=5, filter_height=3, n_filters=4, strides=1,
                padding='VALID')

            tensor = nn.Input(shape=shape)
            out1 = layer1(tensor)
            layer2.set_parameter_variables(
                filter=layer1.get_parameter_variable('filter'),
                bias=layer1.get_parameter_variable('bias'))
            out2 = layer2(tensor)

        for key in ['filter', 'bias']:
            var1 = layer1.get_parameter_variable(key)
            var2 = layer2.get_parameter_variable(key)
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
