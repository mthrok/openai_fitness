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


class TestFlatten(TestCase):
    """Test Flatten layer"""
    def test_flatten(self):
        """4D input tensor is flattened to 2D tensor"""
        input_shape = (32, 4, 77, 84)
        input_value = np.random.rand(*input_shape)
        scope = self.get_scope()
        with nn.variable_scope(scope, reuse=False):
            input_tensor = nn.Input(shape=input_shape)
            flatten = nn.layer.Flatten()
            output_tensor = flatten(input_tensor)

        session = nn.Session()
        output_value = session.run(
            outputs=output_tensor, inputs={input_tensor: input_value})

        self.assertEqual(output_value.shape, output_tensor.shape)
        expected = input_value.reshape(input_value.shape[0], -1)
        np.testing.assert_almost_equal(expected, output_value)


class TestConcat(TestCase):
    """Test Concat Layer behavior"""
    def test_concate_2d_axis_1(self):
        """Concatenate 2 2D tensors"""
        axis, shape1, shape2 = 1, (2, 5), (2, 3)
        with nn.variable_scope(self.get_scope(), reuse=False):
            var1 = nn.make_variable(name='name1', shape=shape1)
            var2 = nn.make_variable(name='name2', shape=shape2)
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
            var1 = nn.make_variable(name='var1', shape=shape1)
            var2 = nn.make_variable(name='var2', shape=shape2)
            var3 = nn.make_variable(name='var3', shape=shape3)
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
