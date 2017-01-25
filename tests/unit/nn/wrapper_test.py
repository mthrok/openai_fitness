from __future__ import absolute_import

import unittest

import numpy as np

from luchador import nn

from tests.unit import fixture


class TestTensorOps(unittest.TestCase):
    def test_mul_numbers(self):
        constant, shape = 10, (3, 5)
        with nn.variable_scope(self.id().replace('.', '/')):
            tensor1 = fixture.create_tensor(shape, dtype='int32')
            tensor2 = tensor1 * constant
            tensor3 = constant * tensor1

        session = nn.Session()

        val1, val2, val3 = session.run(
            outputs=[tensor1, tensor2, tensor3],
        )
        self.assertTrue(np.all(val1 * constant == val2))
        self.assertTrue(np.all(constant * val1 == val3))

    def test_mul_tensor(self):
        shape = (3, 5)
        with nn.variable_scope(self.id().replace('.', '/')):
            tensor1 = fixture.create_tensor(shape, dtype='int32')
            tensor2 = tensor1 * tensor1

        session = nn.Session()

        val1, val2 = session.run(
            outputs=[tensor1, tensor2],
        )
        self.assertTrue(np.all(val1 * val1 == val2))

    def test_mul_variable(self):
        shape = (3, 5)
        with nn.variable_scope(self.id().replace('.', '/')):
            tensor1 = fixture.create_tensor(shape, dtype='int32')
            variable = fixture.create_variable(shape, dtype='int32')
            tensor2 = tensor1 * variable
            tensor3 = variable * tensor1

        session = nn.Session()
        session.initialize()

        val1, val2, val3, val0 = session.run(
            outputs=[tensor1, tensor2, tensor3, variable],
        )
        self.assertTrue(np.all(val1 * val0 == val2))
        self.assertTrue(np.all(val0 * val1 == val3))

    def test_mul_input(self):
        shape = (3, 5)
        with nn.variable_scope(self.id().replace('.', '/')):
            tensor1 = fixture.create_tensor(shape, dtype='int32')
            input_ = nn.Input(name='input', shape=shape, dtype='int32')
            tensor2 = tensor1 * input_
            tensor3 = input_ * tensor1

        session = nn.Session()
        session.initialize()

        input_val = 5 * np.ones(shape, dtype='int32')
        val1, val2, val3 = session.run(
            outputs=[tensor1, tensor2, tensor3],
            givens={input_: input_val}
        )
        self.assertTrue(np.all(val1 * input_val == val2))
        self.assertTrue(np.all(input_val * val1 == val3))

    def test_add_numbers(self):
        constant, shape = 10, (3, 5)
        with nn.variable_scope(self.id().replace('.', '/')):
            tensor1 = fixture.create_tensor(shape, dtype='int32')
            tensor2 = tensor1 + constant
            tensor3 = constant + tensor1

        session = nn.Session()

        val1, val2, val3 = session.run(
            outputs=[tensor1, tensor2, tensor3],
        )
        self.assertTrue(np.all(val1 + constant == val2))
        self.assertTrue(np.all(constant + val1 == val3))

    def test_add_tensor(self):
        shape = (3, 5)
        with nn.variable_scope(self.id().replace('.', '/')):
            tensor1 = fixture.create_tensor(shape, dtype='int32')
            tensor2 = tensor1 + tensor1

        session = nn.Session()
        val1, val2 = session.run(
            outputs=[tensor1, tensor2],
        )
        self.assertTrue(np.all(val1 + val1 == val2))

    def test_add_input(self):
        shape = (3, 5)
        with nn.variable_scope(self.id().replace('.', '/')):
            input_ = nn.Input(shape=shape, dtype='int32')
            tensor1 = fixture.create_tensor(shape, dtype='int32')
            tensor2 = tensor1 + input_
            tensor3 = input_ + tensor1

        session = nn.Session()
        val0 = 10 * np.ones(shape, dtype='int32')
        val1, val2, val3 = session.run(
            outputs=[tensor1, tensor2, tensor3],
            givens={input_: val0}
        )
        self.assertTrue(np.all(val1 + val0 == val2))
        self.assertTrue(np.all(val0 + val1 == val3))

    def test_add_variable(self):
        shape = (3, 5)
        with nn.variable_scope(self.id().replace('.', '/')):
            tensor1 = fixture.create_tensor(shape, dtype='int32')
            variable = fixture.create_variable(shape, dtype='int32')
            tensor2 = tensor1 + variable
            tensor3 = variable + tensor1

        session = nn.Session()
        session.initialize()

        val1, val2, val3, val0 = session.run(
            outputs=[tensor1, tensor2, tensor3, variable],
        )
        self.assertTrue(np.all(val1 + val0 == val2))
        self.assertTrue(np.all(val0 + val1 == val3))

    def test_neg(self):
        shape = (3, 5)
        with nn.variable_scope(self.id().replace('.', '/')):
            tensor1 = fixture.create_tensor(shape, dtype='int32')
            tensor2 = -tensor1

        session = nn.Session()
        val1, val2 = session.run(
            outputs=[tensor1, tensor2],
        )
        self.assertTrue(np.all(-val1 == val2))

    def test_sub_numbers(self):
        constant, shape = 10, (3, 5)
        with nn.variable_scope(self.id().replace('.', '/')):
            tensor1 = fixture.create_tensor(shape, dtype='int32')
            tensor2 = tensor1 - constant
            tensor3 = constant - tensor1

        session = nn.Session()

        val1, val2, val3 = session.run(
            outputs=[tensor1, tensor2, tensor3],
        )
        self.assertTrue(np.all(val1 - constant == val2))
        self.assertTrue(np.all(constant - val1 == val3))

    def test_sub_tensor(self):
        shape = (3, 5)
        with nn.variable_scope(self.id().replace('.', '/')):
            tensor1 = fixture.create_tensor(shape, dtype='int32')
            tensor2 = tensor1 * tensor1

        session = nn.Session()

        val1, val2 = session.run(
            outputs=[tensor1, tensor2],
        )
        self.assertTrue(np.all(val1 * val1 == val2))

    def test_sub_input(self):
        shape = (3, 5)
        with nn.variable_scope(self.id().replace('.', '/')):
            input_ = nn.Input(shape=shape, dtype='int32')
            tensor1 = fixture.create_tensor(shape, dtype='int32')
            tensor2 = tensor1 - input_
            tensor3 = input_ - tensor1

        session = nn.Session()
        val0 = 10 * np.ones(shape, dtype='int32')
        val1, val2, val3 = session.run(
            outputs=[tensor1, tensor2, tensor3],
            givens={input_: val0}
        )
        self.assertTrue(np.all(val1 - val0 == val2))
        self.assertTrue(np.all(val0 - val1 == val3))

    def test_sub_variable(self):
        shape = (3, 5)
        with nn.variable_scope(self.id().replace('.', '/')):
            tensor1 = fixture.create_tensor(shape, dtype='int32')
            variable = fixture.create_variable(shape, dtype='int32')
            tensor2 = tensor1 - variable
            tensor3 = variable - tensor1

        session = nn.Session()
        session.initialize()

        val1, val2, val3, val0 = session.run(
            outputs=[tensor1, tensor2, tensor3, variable],
        )
        self.assertTrue(np.all(val1 - val0 == val2))
        self.assertTrue(np.all(val0 - val1 == val3))

    def _test_mean(self, axis, shape, keep_dims):
        with nn.variable_scope(self.id().replace('.', '/')):
            tensor0 = fixture.create_tensor(shape, dtype='float32')
            tensor1 = tensor0.mean(axis=axis, keep_dims=keep_dims)

        session = nn.Session()

        val0, val1 = session.run(
            outputs=[tensor0, tensor1],
        )
        expected = val0.mean(axis=axis, keepdims=keep_dims)
        self.assertTrue(np.all(val1 == expected))

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
