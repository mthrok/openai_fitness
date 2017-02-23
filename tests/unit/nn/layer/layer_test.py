"""Test Layer behaviors"""
from __future__ import division
from __future__ import absolute_import

import numpy as np

# import theano
# theano.config.optimizer = 'None'
# theano.config.exception_verbosity = 'high'

from luchador import nn
from tests.unit.fixture import TestCase

# pylint: disable=invalid-name,too-many-locals


class TestDense(TestCase):
    """Test Dense layer"""
    def test_dynamic_initializer(self):
        """Initializers are correctly selected"""
        n_in, n_nodes, weight_val, bias_val = 4, 5, 13, 7
        with nn.variable_scope(self.get_scope()):
            dense = nn.layer.Dense(
                n_nodes=5,
                initializers={
                    'weight': {
                        'typename': 'Constant',
                        'args': {
                            'value': weight_val,
                        },
                    },
                    'bias': {
                        'typename': 'Constant',
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
            outputs=dense.get_parameter_variables()
        )

        np.testing.assert_almost_equal(
            weight, weight_val * np.ones((n_in, n_nodes)))
        np.testing.assert_almost_equal(
            bias, bias_val * np.ones((n_nodes,)))
