from __future__ import division
from __future__ import absolute_import

import unittest

import numpy as np
# import theano
# theano.config.optimizer = 'None'
# theano.config.exception_verbosity = 'high'

import luchador
from luchador.nn import (
    Input,
    Session,
    BatchNormalization,
    NCHW2NHWC,
    NHWC2NCHW,
    scope as scp
)


BE = luchador.get_nn_backend()


def is_gpu_available():
    from tensorflow.python.client import device_lib
    for x in device_lib.list_local_devices():
        if x.device_type == 'GPU':
            return True
    return False


class BatchNormalizationTest(unittest.TestCase):
    """Test for BatchNormalization class"""
    def setUp(self):
        self.conv_format = luchador.get_nn_conv_format()

    def tearDown(self):
        luchador.set_nn_conv_format(self.conv_format)

    def _normalize(self, shape, offset, scale):
        _shape = (None,) + shape[1:]
        input_tensor = Input(shape=_shape).build()
        input_value = np.random.randn(*shape) - 100

        bn = BatchNormalization(
            scale=scale, offset=offset, learn=True, decay=0.0)
        normalized = bn(input_tensor)
        updates = bn.get_update_operation()
        session = Session()
        session.initialize()
        return session.run(
            outputs=normalized,
            inputs={input_tensor: input_value}, updates=updates)

    def test_normalization_2d(self):
        """Output of normalization layer is normalized on 2D array"""
        offset, scale, shape = 10.0, 1.0, (64, 16)

        with scp.variable_scope(self.id().replace('.', '/')):
            output_value = self._normalize(shape, offset, scale)

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

    @unittest.skipIf(BE == 'tensorflow' and not is_gpu_available(),
                     'Skipping as no GPU is found in TF backend')
    def test_normalization_4d_NCHW(self):
        """Output of normalization layer is normalized on 4D array"""
        luchador.set_nn_conv_format('NCHW')
        offset, scale, shape = 3.0, 7.0, (32, 16, 8, 7)
        with scp.variable_scope(self.id().replace('.', '/')):
            output_value = self._normalize(shape, offset, scale)

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
        with scp.variable_scope(self.id().replace('.', '/')):
            output_value = self._normalize(shape, offset, scale)

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
        input_tensor = Input(shape=[None, shape[1]]).build()

        bn = BatchNormalization(learn=True, decay=0.999)
        normalized = bn(input_tensor)

        mean_tensor = bn.parameter_variables['mean']
        var_tensor = bn.parameter_variables['var']
        updates = bn.get_update_operation()

        input_value = np.random.randn(*shape) - 100
        true_mean = input_value.mean(axis=0)
        true_var = input_value.var(axis=0)

        session = Session()
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


def _convert(layer, shape):
    input_tensor = Input(shape=shape).build()
    input_value = np.random.randn(*shape) - 100

    session = Session()

    output_tensor = layer(input_tensor)
    output_value = session.run(
        outputs=output_tensor,
        inputs={input_tensor: input_value},
    )
    return output_value, output_tensor


class FormatConversionTest(unittest.TestCase):
    def test_NCHW2NHWC(self):
        shape = (32, 4, 7, 8)
        with scp.variable_scope(self.id().replace('.', '/')):
            output_value, output_tensor = _convert(NCHW2NHWC(), shape)

        expected = (shape[0], shape[2], shape[3], shape[1])
        self.assertEqual(expected, output_value.shape)
        self.assertEqual(expected, output_tensor.shape)

    def test_NHWC2NCHW(self):
        shape = (32, 8, 7, 4)
        with scp.variable_scope(self.id().replace('.', '/')):
            output_value, output_tensor = _convert(NHWC2NCHW(), shape)

        expected = (shape[0], shape[3], shape[1], shape[2])
        self.assertEqual(expected, output_value.shape)
        self.assertEqual(expected, output_tensor.shape)
