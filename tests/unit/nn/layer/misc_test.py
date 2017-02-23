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


def _convert(layer, shape):
    input_tensor = nn.Input(shape=shape)
    input_value = np.random.randn(*shape) - 100

    session = nn.Session()

    output_tensor = layer(input_tensor)
    output_value = session.run(
        outputs=output_tensor,
        inputs={input_tensor: input_value},
    )
    return output_value, output_tensor


class FormatConversionTest(TestCase):
    """Test conversion layer"""
    def test_NCHW2NHWC(self):
        """Test NCHW to NHWC conversion"""
        shape = (32, 4, 7, 8)
        with nn.variable_scope(self.get_scope()):
            output_value, output_tensor = _convert(
                nn.layer.NCHW2NHWC(), shape)

        expected = (shape[0], shape[2], shape[3], shape[1])
        self.assertEqual(expected, output_value.shape)
        self.assertEqual(expected, output_tensor.shape)

    def test_NHWC2NCHW(self):
        """Test NHWC to NCHW conversion"""
        shape = (32, 8, 7, 4)
        with nn.variable_scope(self.get_scope()):
            output_value, output_tensor = _convert(
                nn.layer.NHWC2NCHW(), shape)

        expected = (shape[0], shape[3], shape[1], shape[2])
        self.assertEqual(expected, output_value.shape)
        self.assertEqual(expected, output_tensor.shape)
