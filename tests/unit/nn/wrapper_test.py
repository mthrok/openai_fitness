"""Test Wapper methods"""
from __future__ import absolute_import

import unittest

import numpy as np

from luchador import nn

from tests.unit import fixture


class TestTensorOps(unittest.TestCase):
    """Test wrapper operations"""
    def test_mul_numbers(self):
        """Tensor * number is correct elementwise"""
        constant, shape = 10, (3, 5)
        with nn.variable_scope(self.id().replace('.', '/')):
            tensor1 = fixture.create_ones_tensor(shape, dtype='int32')
            tensor2 = tensor1 * constant
            tensor3 = constant * tensor1

        session = nn.Session()

        val1, val2, val3 = session.run(
            outputs=[tensor1, tensor2, tensor3],
        )
        np.testing.assert_equal(val1 * constant, val2)
        np.testing.assert_equal(constant * val1, val3)

    def test_mul_tensor(self):
        """Tensor * Tensor is correct elementwise"""
        shape = (3, 5)
        with nn.variable_scope(self.id().replace('.', '/')):
            tensor1 = fixture.create_ones_tensor(shape, dtype='int32')
            tensor2 = tensor1 * tensor1

        session = nn.Session()

        val1, val2 = session.run(
            outputs=[tensor1, tensor2],
        )
        np.testing.assert_equal(val1 * val1, val2)

    def test_mul_variable(self):
        """Tensor * Variable is correct elementwise"""
        shape = (3, 5)
        with nn.variable_scope(self.id().replace('.', '/')):
            tensor1 = fixture.create_ones_tensor(shape, dtype='int32')
            variable = fixture.create_constant_variable(shape, dtype='int32')
            tensor2 = tensor1 * variable
            tensor3 = variable * tensor1

        session = nn.Session()
        session.initialize()

        val1, val2, val3, val0 = session.run(
            outputs=[tensor1, tensor2, tensor3, variable],
        )
        np.testing.assert_equal(val1 * val0, val2)
        np.testing.assert_equal(val0 * val1, val3)

    def test_mul_input(self):
        """Tensor * Input is correct elementwise"""
        shape = (3, 5)
        with nn.variable_scope(self.id().replace('.', '/')):
            tensor1 = fixture.create_ones_tensor(shape, dtype='int32')
            input_ = nn.Input(name='input', shape=[], dtype='int32')
            tensor2 = tensor1 * input_
            tensor3 = input_ * tensor1

        session = nn.Session()
        session.initialize()

        input_val = np.array(5, dtype='int32')
        val1, val2, val3 = session.run(
            outputs=[tensor1, tensor2, tensor3],
            givens={input_: input_val}
        )
        np.testing.assert_equal(val1 * input_val, val2)
        np.testing.assert_equal(input_val * val1, val3)

    def test_truediv_numbers(self):
        """Tensor / number is correct elementwise"""
        constant, shape = 10., (3, 5)
        with nn.variable_scope(self.id().replace('.', '/')):
            tensor1 = fixture.create_ones_tensor(shape, dtype='float32')
            tensor2 = tensor1 / constant
            tensor3 = constant / tensor1

        session = nn.Session()

        val1, val2, val3 = session.run(
            outputs=[tensor1, tensor2, tensor3],
        )
        np.testing.assert_equal(np.true_divide(val1, constant), val2)
        np.testing.assert_equal(np.true_divide(constant, val1), val3)

    def test_floordiv_numbers(self):
        """Tensor // number is correct elementwise"""
        constant, shape = 10., (3, 5)
        with nn.variable_scope(self.id().replace('.', '/')):
            tensor1 = fixture.create_ones_tensor(shape, dtype='float32')
            tensor2 = tensor1 // constant
            tensor3 = constant // tensor1

        session = nn.Session()

        val1, val2, val3 = session.run(
            outputs=[tensor1, tensor2, tensor3],
        )
        np.testing.assert_equal(np.floor_divide(val1, constant), val2)
        np.testing.assert_equal(np.floor_divide(constant, val1), val3)

    def test_truediv_tensor(self):
        """Tensor / Tensor is correct elementwise"""
        shape = (3, 5)
        with nn.variable_scope(self.id().replace('.', '/')):
            tensor1 = fixture.create_ones_tensor(shape, dtype='float32')
            tensor2 = tensor1 / tensor1

        session = nn.Session()

        val1, val2 = session.run(
            outputs=[tensor1, tensor2],
        )
        np.testing.assert_equal(np.true_divide(val1, val1), val2)

    def test_floordiv_tensor(self):
        """Tensor // Tensor is correct elementwise"""
        shape = (3, 5)
        with nn.variable_scope(self.id().replace('.', '/')):
            tensor1 = fixture.create_ones_tensor(shape, dtype='float32')
            tensor2 = tensor1 // tensor1

        session = nn.Session()

        val1, val2 = session.run(
            outputs=[tensor1, tensor2],
        )
        np.testing.assert_equal(np.floor_divide(val1, val1), val2)

    def test_truediv_input(self):
        """Tensor / Input is correct elementwise"""
        constant, shape = 10., (3, 5)
        with nn.variable_scope(self.id().replace('.', '/')):
            tensor1 = fixture.create_ones_tensor(shape, dtype='float32')
            input1 = nn.Input(shape=[], dtype='float32')
            tensor2 = tensor1 / input1
            tensor3 = input1 / tensor1

        session = nn.Session()
        session.initialize()

        val1, val2, val3 = session.run(
            outputs=[tensor1, tensor2, tensor3],
            givens={input1: np.array(constant, dtype='float32')}
        )
        np.testing.assert_equal(np.true_divide(val1, constant), val2)
        np.testing.assert_equal(np.true_divide(constant, val1), val3)

    def test_floordiv_input(self):
        """Tensor // Input is correct elementwise"""
        constant, shape = 10., (3, 5)
        with nn.variable_scope(self.id().replace('.', '/')):
            tensor1 = fixture.create_ones_tensor(shape, dtype='float32')
            input1 = nn.Input(shape=[], dtype='float32')
            tensor2 = tensor1 // input1
            tensor3 = input1 // tensor1

        session = nn.Session()
        session.initialize()

        val1, val2, val3 = session.run(
            outputs=[tensor1, tensor2, tensor3],
            givens={input1: np.array(constant, dtype='float32')}
        )
        np.testing.assert_equal(np.floor_divide(val1, constant), val2)
        np.testing.assert_equal(np.floor_divide(constant, val1), val3)

    def test_add_numbers(self):
        """Tensor + number is correct elementwise"""
        constant, shape = 10, (3, 5)
        with nn.variable_scope(self.id().replace('.', '/')):
            tensor1 = fixture.create_ones_tensor(shape, dtype='int32')
            tensor2 = tensor1 + constant
            tensor3 = constant + tensor1

        session = nn.Session()

        val1, val2, val3 = session.run(
            outputs=[tensor1, tensor2, tensor3],
        )
        np.testing.assert_equal(val1 + constant, val2)
        np.testing.assert_equal(constant + val1, val3)

    def test_add_tensor(self):
        """Tensor + Tensor is correct elementwise"""
        shape = (3, 5)
        with nn.variable_scope(self.id().replace('.', '/')):
            tensor1 = fixture.create_ones_tensor(shape, dtype='int32')
            tensor2 = tensor1 + tensor1

        session = nn.Session()
        val1, val2 = session.run(
            outputs=[tensor1, tensor2],
        )
        np.testing.assert_equal(val1 + val1, val2)

    def test_add_input(self):
        """Tensor + Input is correct elementwise"""
        shape = (3, 5)
        with nn.variable_scope(self.id().replace('.', '/')):
            input_ = nn.Input(shape=[], dtype='int32')
            tensor1 = fixture.create_ones_tensor(shape, dtype='int32')
            tensor2 = tensor1 + input_
            tensor3 = input_ + tensor1

        session = nn.Session()
        val0 = np.array(10, dtype='int32')
        val1, val2, val3 = session.run(
            outputs=[tensor1, tensor2, tensor3],
            givens={input_: val0}
        )
        np.testing.assert_equal(val1 + val0, val2)
        np.testing.assert_equal(val0 + val1, val3)

    def test_add_variable(self):
        """Tensor + Variable is correct elementwise"""
        shape = (3, 5)
        with nn.variable_scope(self.id().replace('.', '/')):
            tensor1 = fixture.create_ones_tensor(shape, dtype='int32')
            variable = fixture.create_constant_variable(shape, dtype='int32')
            tensor2 = tensor1 + variable
            tensor3 = variable + tensor1

        session = nn.Session()
        session.initialize()

        val1, val2, val3, val0 = session.run(
            outputs=[tensor1, tensor2, tensor3, variable],
        )
        np.testing.assert_equal(val1 + val0, val2)
        np.testing.assert_equal(val0 + val1, val3)

    def test_neg(self):
        """-Tensor is correct elementwise"""
        shape = (3, 5)
        with nn.variable_scope(self.id().replace('.', '/')):
            tensor1 = fixture.create_ones_tensor(shape, dtype='int32')
            tensor2 = -tensor1

        session = nn.Session()
        val1, val2 = session.run(
            outputs=[tensor1, tensor2],
        )
        np.testing.assert_equal(-val1, val2)

    def test_sub_numbers(self):
        """Tensor - number is correct elementwise"""
        constant, shape = 10, (3, 5)
        with nn.variable_scope(self.id().replace('.', '/')):
            tensor1 = fixture.create_ones_tensor(shape, dtype='int32')
            tensor2 = tensor1 - constant
            tensor3 = constant - tensor1

        session = nn.Session()

        val1, val2, val3 = session.run(
            outputs=[tensor1, tensor2, tensor3],
        )
        np.testing.assert_equal(val1 - constant, val2)
        np.testing.assert_equal(constant - val1, val3)

    def test_sub_tensor(self):
        """Tensor - Tensor is correct elementwise"""
        shape = (3, 5)
        with nn.variable_scope(self.id().replace('.', '/')):
            tensor1 = fixture.create_ones_tensor(shape, dtype='int32')
            tensor2 = tensor1 - tensor1

        session = nn.Session()

        val1, val2 = session.run(
            outputs=[tensor1, tensor2],
        )
        np.testing.assert_equal(val1 - val1, val2)

    def test_sub_input(self):
        """Tensor - Input is correct elementwise"""
        shape = (3, 5)
        with nn.variable_scope(self.id().replace('.', '/')):
            input_ = nn.Input(shape=[], dtype='int32')
            tensor1 = fixture.create_ones_tensor(shape, dtype='int32')
            tensor2 = tensor1 - input_
            tensor3 = input_ - tensor1

        session = nn.Session()
        val0 = np.array(10, dtype='int32')
        val1, val2, val3 = session.run(
            outputs=[tensor1, tensor2, tensor3],
            givens={input_: val0}
        )
        np.testing.assert_equal(val1 - val0, val2)
        np.testing.assert_equal(val0 - val1, val3)

    def test_sub_variable(self):
        """Tensor - Variable is correct elementwise"""
        shape = (3, 5)
        with nn.variable_scope(self.id().replace('.', '/')):
            tensor1 = fixture.create_ones_tensor(shape, dtype='int32')
            variable = fixture.create_constant_variable(shape, dtype='int32')
            tensor2 = tensor1 - variable
            tensor3 = variable - tensor1

        session = nn.Session()
        session.initialize()

        val1, val2, val3, val0 = session.run(
            outputs=[tensor1, tensor2, tensor3, variable],
        )
        np.testing.assert_equal(val1 - val0, val2)
        np.testing.assert_equal(val0 - val1, val3)

    def _test_mean(self, axis, shape, keep_dims):
        with nn.variable_scope(self.id().replace('.', '/')):
            tensor0 = fixture.create_ones_tensor(shape, dtype='float32')
            tensor1 = tensor0.mean(axis=axis, keep_dims=keep_dims)

        session = nn.Session()

        val0, val1 = session.run(
            outputs=[tensor0, tensor1],
        )
        expected = val0.mean(axis=axis, keepdims=keep_dims)
        np.testing.assert_equal(val1, expected)

    def test_mean(self):
        """Test mean with single axis, dropping axis"""
        self._test_mean(0, (3, 5), False)

    def test_mean_keep_dim(self):
        """Test mean with single axis, dropping axis"""
        self._test_mean(0, (3, 5), True)

    def test_mean_multi(self):
        """Test mean with multiple axes, dropping axis"""
        self._test_mean((1, 2), (3, 4, 5, 6), False)

    def test_mean_multi_keep_dim(self):
        """Test mean with multiple axes, dropping axis"""
        self._test_mean((1, 2), (3, 4, 5, 6), True)

    def _test_max(self, axis, shape, keep_dims):
        with nn.variable_scope(self.id().replace('.', '/')):
            tensor0 = fixture.create_ones_tensor(shape, dtype='float32')
            tensor1 = tensor0.max(axis=axis, keep_dims=keep_dims)

        session = nn.Session()

        val0, val1 = session.run(
            outputs=[tensor0, tensor1],
        )
        expected = val0.max(axis=axis, keepdims=keep_dims)
        np.testing.assert_equal(val1, expected)

    def test_max(self):
        """Test max with single axis, dropping axis"""
        self._test_max(0, (3, 5), False)

    def test_max_keep_dim(self):
        """Test max with single axis, dropping axis"""
        self._test_max(0, (3, 5), True)

    def test_max_multi(self):
        """Test max with multiple axes, dropping axis"""
        self._test_max((1, 2), (3, 4, 5, 6), False)

    def test_max_multi_keep_dim(self):
        """Test max with multiple axes, dropping axis"""
        self._test_max((1, 2), (3, 4, 5, 6), True)

    def test_clip_number(self):
        """Test clip with float"""
        shape, min_value, max_value = (10, 10), 0.4, 0.6
        with nn.variable_scope(self.id().replace('.', '/')):
            variable0 = fixture.create_random_variable(shape, dtype='float32')
            tensor1 = variable0.clip(max_value=max_value, min_value=min_value)

        session = nn.Session()
        session.initialize()

        val0, val1 = session.run(
            outputs=[variable0, tensor1],
        )
        expected = np.clip(val0, a_max=max_value, a_min=min_value)
        np.testing.assert_almost_equal(val1, expected)

    def test_clip_variable(self):
        """Test clip with Variable"""
        shape, min_value, max_value = (10, 10), 0.4, 0.6
        with nn.variable_scope(self.id().replace('.', '/')):
            variable0 = fixture.create_random_variable(shape, dtype='float32')
            min_variable = fixture.create_constant_variable(
                shape=[], dtype='float32', value=min_value, name='min_var')
            max_variable = fixture.create_constant_variable(
                shape=[], dtype='float32', value=max_value, name='max_var')
            tensor1 = variable0.clip(
                max_value=max_variable, min_value=min_variable)

        session = nn.Session()
        session.initialize()

        val0, val1 = session.run(
            outputs=[variable0, tensor1],
        )
        expected = np.clip(val0, a_max=max_value, a_min=min_value)
        np.testing.assert_almost_equal(val1, expected)

    def test_clip_tensor(self):
        """Test clip with Tensor"""
        shape, min_value, max_value = (10, 10), 0.4, 0.6
        with nn.variable_scope(self.id().replace('.', '/')):
            variable0 = fixture.create_random_variable(shape, dtype='float32')
            min_tensor = min_value * fixture.create_ones_tensor(
                shape=[], dtype='float32', name='min_tensor')
            max_tensor = max_value * fixture.create_ones_tensor(
                shape=[], dtype='float32', name='max_tensor')
            tensor1 = variable0.clip(
                max_value=max_tensor, min_value=min_tensor)

        session = nn.Session()
        session.initialize()

        val0, val1 = session.run(
            outputs=[variable0, tensor1],
        )
        expected = np.clip(val0, a_max=max_value, a_min=min_value)
        np.testing.assert_almost_equal(val1, expected)

    def test_clip_input(self):
        """Test clip with Input"""
        shape, min_value, max_value = (10, 10), 0.4, 0.6
        with nn.variable_scope(self.id().replace('.', '/')):
            variable0 = fixture.create_random_variable(shape, dtype='float32')
            min_input = nn.Input(shape=[], dtype='float32')
            max_input = nn.Input(shape=[], dtype='float32')
            tensor1 = variable0.clip(
                max_value=max_input, min_value=min_input)

        session = nn.Session()
        session.initialize()

        val0, val1 = session.run(
            outputs=[variable0, tensor1],
            givens={
                min_input: np.array(min_value, dtype='float32'),
                max_input: np.array(max_value, dtype='float32'),
            },
        )
        expected = np.clip(val0, a_max=max_value, a_min=min_value)
        np.testing.assert_almost_equal(val1, expected)
