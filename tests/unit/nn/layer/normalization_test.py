"""Test Layer behaviors"""
from __future__ import division
from __future__ import absolute_import

import unittest

import numpy as np

# import theano
# theano.config.optimizer = 'None'
# theano.config.exception_verbosity = 'high'

import luchador
from luchador import nn
from tests.unit.fixture import TestCase

BE = luchador.get_nn_backend()

# pylint: disable=invalid-name,too-many-locals


def _is_gpu_available():
    from tensorflow.python.client import device_lib
    for x in device_lib.list_local_devices():
        if x.device_type == 'GPU':
            return True
    return False


def _normalize_batch(shape, offset, scale):
    _shape = (None,) + shape[1:]
    input_tensor = nn.Input(shape=_shape)
    input_value = np.random.randn(*shape) - 100

    bn = nn.layer.BatchNormalization(
        scale=scale, offset=offset, learn=True, decay=0.0)
    normalized = bn(input_tensor)
    updates = bn.get_update_operations()
    session = nn.Session()
    session.initialize()
    return session.run(
        outputs=normalized,
        inputs={input_tensor: input_value}, updates=updates)


class BatchNormalizationTest(TestCase):
    """Test for BatchNormalization class"""
    def setUp(self):
        self.conv_format = luchador.get_nn_conv_format()

    def tearDown(self):
        luchador.set_nn_conv_format(self.conv_format)

    def test_normalization_2d(self):
        """Output of normalization layer is normalized on 2D array"""
        offset, scale, shape = 10.0, 1.0, (64, 16)

        with nn.variable_scope(self.id().replace('.', '/')):
            output_value = _normalize_batch(shape, offset, scale)

        self.assertEqual(output_value.shape, shape)

        for c in range(shape[1]):
            column = output_value[:, c]

            expected = offset
            found = column.mean()
            diff = abs(expected - found) / expected
            threshold = 0.01
            self.assertTrue(
                diff < threshold,
                'The mean value of column {} must be close enough to '
                'the target offset value. Expected: {}, Found: {}'
                .format(c, expected, found)
            )

            expected = scale
            found = column.std()
            diff = abs(expected - found) / expected
            threshold = 0.01
            self.assertTrue(
                diff < threshold,
                'The variance of column {} must be close enough to '
                'the target offset value. Expected: {}, Found: {}'
                .format(c, expected, found)
            )

    @unittest.skipIf(BE == 'tensorflow' and not _is_gpu_available(),
                     'Skipping as no GPU is found in TF backend')
    def test_normalization_4d_NCHW(self):
        """Output of normalization layer is normalized on 4D array"""
        luchador.set_nn_conv_format('NCHW')
        offset, scale, shape = 3.0, 7.0, (32, 16, 8, 7)
        with nn.variable_scope(self.get_scope()):
            output_value = _normalize_batch(shape, offset, scale)

        self.assertEqual(output_value.shape, shape)

        for c in range(shape[1]):
            channel = output_value[:, c]

            expected = offset
            found = channel.mean()
            diff = abs(expected - found) / expected
            threshold = 0.01
            self.assertTrue(
                diff < threshold,
                'The mean value of channel {} must be close enough to '
                'the target offset value. Expected: {}, Found: {}'
                .format(c, expected, found)
            )

            expected = scale
            found = channel.std()
            diff = abs(expected - found) / expected
            threshold = 0.01
            self.assertTrue(
                diff < threshold,
                'The variance of channel {} must be close enough to '
                'the target offset value. Expected: {}, Found: {}'
                .format(c, expected, found)
            )

    @unittest.skipUnless(BE == 'tensorflow', 'Tensorflow only')
    def test_normalization_4d_NHWC(self):
        """Output of normalization layer is normalized on 4D array"""
        luchador.set_nn_conv_format('NHWC')
        offset, scale, shape = 3.0, 7.0, (32, 8, 7, 16)
        with nn.variable_scope(self.get_scope()):
            output_value = _normalize_batch(shape, offset, scale)

        self.assertEqual(output_value.shape, shape)

        for c in range(shape[3]):
            channel = output_value[:, :, :, c]

            expected = offset
            found = channel.mean()
            diff = abs(expected - found) / expected
            threshold = 0.01
            self.assertTrue(
                diff < threshold,
                'The mean value of channel {} must be close enough to '
                'the target offset value. Expected: {}, Found: {}'
                .format(c, expected, found)
            )

            expected = scale
            found = channel.std()
            diff = abs(expected - found) / expected
            threshold = 0.03
            self.assertTrue(
                diff < threshold,
                'The variance of channel {} must be close enough to '
                'the target offset value. Expected: {}, Found: {}'
                .format(c, expected, found)
            )

    def test_check_stats_regression(self):
        """Layer mean and std regresses to those of sample batch"""
        shape = (32, 5)
        input_tensor = nn.Input(shape=[None, shape[1]])

        scope = self.get_scope()
        with luchador.nn.variable_scope(scope, reuse=False):
            bn = nn.layer.BatchNormalization(learn=True, decay=0.999)
            normalized = bn(input_tensor)

        mean_tensor = bn.get_parameter_variable('mean')
        var_tensor = bn.get_parameter_variable('var')
        updates = bn.get_update_operations()

        input_value = np.random.randn(*shape) - 100
        true_mean = input_value.mean(axis=0)
        true_var = input_value.var(axis=0)

        session = nn.Session()
        session.initialize()
        mean_diff_prev, stdi_diff_prev = None, None
        for i in range(30):
            _, mean_val, var_val = session.run(
                outputs=[normalized, mean_tensor, var_tensor],
                inputs={input_tensor: input_value},
                updates=updates,
                name='run'
            )

            mean_diff = abs(true_mean - mean_val)
            var_diff = abs(true_var - var_val)

            if i > 0:
                self.assertTrue(
                    np.all(mean_diff < mean_diff_prev),
                    'Layer mean value is not regressing to the sample mean')
                self.assertTrue(
                    np.all(var_diff < stdi_diff_prev),
                    'Layer std deviation is not regressing to the batch std')

            mean_diff_prev = mean_diff
            stdi_diff_prev = var_diff
