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
    updates = bn.get_update_operation()
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

        mean_tensor = bn.parameter_variables['mean']
        var_tensor = bn.parameter_variables['var']
        updates = bn.get_update_operation()

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
        shape = (32, 4, 7, 8)
        with nn.variable_scope(self.get_scope()):
            output_value, output_tensor = _convert(
                nn.layer.NCHW2NHWC(), shape)

        expected = (shape[0], shape[2], shape[3], shape[1])
        self.assertEqual(expected, output_value.shape)
        self.assertEqual(expected, output_tensor.shape)

    def test_NHWC2NCHW(self):
        shape = (32, 8, 7, 4)
        with nn.variable_scope(self.get_scope()):
            output_value, output_tensor = _convert(
                nn.layer.NHWC2NCHW(), shape)

        expected = (shape[0], shape[3], shape[1], shape[2])
        self.assertEqual(expected, output_value.shape)
        self.assertEqual(expected, output_tensor.shape)


class TestFlatten(TestCase):
    """Test Flatten layer"""
    def test_flatten(self):
        """4D input tensor is flattened to 2D tensor"""
        input_shape = (32, 4, 77, 84)
        input_value = np.zeros(input_shape)
        scope = self.get_scope()
        with luchador.nn.variable_scope(scope, reuse=False):
            input_tensor = nn.Input(shape=input_shape)
            flatten = nn.layer.Flatten()
            output_tensor = flatten(input_tensor)

        session = nn.Session()
        output_value = session.run(
            outputs=output_tensor, inputs={input_tensor: input_value})

        expected = output_value.shape
        found = output_tensor.shape
        self.assertEqual(expected, found)


class TestTile(TestCase):
    """Test Tile layer"""
    def _test_tile(self, pattern, input_shape, input_value):
        scope = self.get_scope()
        with luchador.nn.variable_scope(scope, reuse=False):
            input_tensor = nn.Input(shape=input_shape)
            tile = nn.layer.Tile(pattern=pattern)
            output_tensor = tile(input_tensor)

        session = nn.Session()
        output_value = session.run(
            outputs=output_tensor, inputs={input_tensor: input_value})

        expected = np.tile(input_value, pattern)
        np.testing.assert_almost_equal(expected, output_value)

    def test_tile_shaped(self):
        """N x 1 is tiled to N x M"""
        value = np.array([i for i in range(32)]).reshape((-1, 1))
        shape, pattern = (32, 1), (1, 5)
        self._test_tile(pattern=pattern, input_shape=shape, input_value=value)

    def test_tile_None(self):
        """None x 1 is tiled to None x M"""
        value = np.array([i for i in range(64)]).reshape((-1, 1))
        shape, pattern = (None, 1), (1, 5)
        self._test_tile(pattern=pattern, input_shape=shape, input_value=value)

    def test_tile(self):
        """[N,] is tiled to N x 1 x M"""
        value = np.array([i for i in range(32)])
        shape, pattern = (32,), (1, 1, 5)
        self._test_tile(pattern=pattern, input_shape=shape, input_value=value)


class TestDense(TestCase):
    """Test Dense layer"""
    def test_recreate_success_with_reuse(self):
        """Copied layer can create node when reuse=True in variable scope"""
        n_nodes = 256
        base_scope = self.get_scope()
        with nn.variable_scope(base_scope, reuse=False):
            input_ = nn.Input(shape=(None, n_nodes), name='foo')
            dense1 = nn.layer.Dense(n_nodes)
            dense1(input_)

            _scope = nn.get_variable_scope()
            with nn.variable_scope(_scope, reuse=True):
                dense2 = nn.layer.Dense(**dense1.serialize()['args'])
                dense2(input_)

        vars1 = dense1.parameter_variables
        vars2 = dense2.parameter_variables
        expected = vars1.keys()
        found = vars2.keys()
        self.assertEqual(
            expected, found,
            'Copied layer should have the same parameter variables '
            'as the original layer. Expected: {}, Found: {}'
            .format(expected, found))
        for key in vars1.keys():
            expected, found = vars1[key], vars2[key]
            self.assertTrue(
                vars1[key].unwrap() is vars2[key].unwrap(),
                'Variable objects in copied layer should be identical '
                'to those in the original layer. Variable {} is not identical.'
                .format(key)
            )

    def test_copy_fail_without_reuse(self):
        """Copied layer fails to create node when reuse is not True"""
        fmt = luchador.get_nn_conv_format()
        shape = (None, 4, 84, 84) if fmt == 'NCHW' else (None, 84, 84, 4)
        base_scope = self.get_scope()
        with nn.variable_scope(base_scope, reuse=False):
            input_ = nn.Input(shape=shape, name='foo')
            conv = nn.layer.Conv2D(84, 84, 4, 4)
            conv(input_)
            try:
                conv2 = nn.layer.Conv2D(**conv.serialize()['args'])
                conv2(input_)
                self.fail('Copied layer should raise ValueError when '
                          'reuse is not enabled in variable scope.')
            except ValueError:
                pass
            except Exception as error:  # pylint: disable=broad-except
                self.fail(
                    'Expected ValueError when copied layer tries to '
                    'create node without reuse enabled in variable scope. '
                    'Found "{}"'.format(error)
                )


class TestConcat(TestCase):
    """Test Concat Layer behavior"""
    def test_concate_2d_axis_1(self):
        """Concatenate 2 2D tensors"""
        axis, shape1, shape2 = 1, (2, 5), (2, 3)
        with nn.variable_scope(self.get_scope(), reuse=False):
            var1 = nn.get_variable(name='name1', shape=shape1)
            var2 = nn.get_variable(name='name2', shape=shape2)
            conc_var = nn.layer.Concat(axis=axis).build([var1, var2])

        session = nn.Session()
        val1, val2 = np.random.rand(*shape1), np.random.rand(*shape2)
        conc_val = session.run(outputs=conc_var, givens={
            var1: val1, var2: val2,
        })

        expected = conc_val.shape
        found = conc_var.shape
        self.assertEqual(found, expected)

        expected = np.concatenate((val1, val2), axis=axis)
        found = conc_val
        np.testing.assert_almost_equal(found, expected)

    def test_concate_2d_axis_1_none(self):
        """Concatenate 2 2D tensors with None"""
        axis, n_elems, shape1, shape2 = 1, 32, (None, 5), (None, 3)
        with nn.variable_scope(self.get_scope(), reuse=False):
            input1 = nn.Input(shape=shape1, dtype='float32', name='name1')
            input2 = nn.Input(shape=shape2, dtype='float32', name='name2')
            conc_var = nn.layer.Concat(axis=axis).build([input1, input2])

        session = nn.Session()
        val1 = np.random.rand(n_elems, shape1[1]).astype('float32')
        val2 = np.random.rand(n_elems, shape2[1]).astype('float32')
        conc_val = session.run(outputs=conc_var, givens={
            input1: val1, input2: val2,
        })

        expected = (None, conc_val.shape[1])
        found = conc_var.shape
        self.assertEqual(found, expected)

        expected = np.concatenate((val1, val2), axis=axis)
        found = conc_val
        np.testing.assert_almost_equal(found, expected)

    def test_concate_2d_axis_1_multiple_none(self):
        """Concatenate 2 2D tensors with None"""
        axis, n_elems, shape1, shape2 = 1, 32, (None, None, 5), (None, 3, None)
        with nn.variable_scope(self.get_scope(), reuse=False):
            input1 = nn.Input(shape=shape1, dtype='float32', name='name1')
            input2 = nn.Input(shape=shape2, dtype='float32', name='name2')
            conc_var = nn.layer.Concat(axis=axis).build([input1, input2])

        session = nn.Session()
        val1 = np.random.rand(n_elems, 4, 5).astype('float32')
        val2 = np.random.rand(n_elems, 3, 5).astype('float32')
        conc_val = session.run(outputs=conc_var, givens={
            input1: val1, input2: val2,
        })
        expected = (None, None, 5)
        found = conc_var.shape
        self.assertEqual(found, expected)

        expected = np.concatenate((val1, val2), axis=axis)
        found = conc_val
        np.testing.assert_almost_equal(found, expected)

    def test_concatenate_raise_when_incosistent_shape(self):
        """Concatenate raise ValueError when inconsistent shapes"""
        axis, shape1, shape2 = 1, (3, 5), (4, 6)
        with nn.variable_scope(self.get_scope(), reuse=False):
            input1 = nn.Input(shape=shape1, dtype='float32', name='name1')
            input2 = nn.Input(shape=shape2, dtype='float32', name='name2')
            with self.assertRaises(ValueError):
                nn.layer.Concat(axis=axis).build([input1, input2])

    def test_concate_2d_axis_1_3(self):
        """Concatenate 3 2D tensors"""
        axis, shape1, shape2, shape3 = 1, (2, 5), (2, 3), (2, 4)
        with nn.variable_scope(self.get_scope(), reuse=False):
            var1 = nn.get_variable(name='var1', shape=shape1)
            var2 = nn.get_variable(name='var2', shape=shape2)
            var3 = nn.get_variable(name='var3', shape=shape3)
            conc_var = nn.layer.Concat(axis=axis).build([var1, var2, var3])

        session = nn.Session()
        val1, val2 = np.random.rand(*shape1), np.random.rand(*shape2)
        val3 = np.random.rand(*shape3)
        conc_val = session.run(outputs=conc_var, givens={
            var1: val1, var2: val2, var3: val3
        })
        expected = conc_val.shape
        found = conc_var.shape
        self.assertEqual(found, expected)

        expected = np.concatenate((val1, val2, val3), axis=axis)
        found = conc_val
        np.testing.assert_almost_equal(found, expected)
