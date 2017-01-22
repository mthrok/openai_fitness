from __future__ import absolute_import

import unittest

import numpy as np

import luchador
from luchador import nn


def _create_ones_tensor(shape):
    if luchador.get_nn_backend() == 'theano':
        import theano.tensor as be
    else:
        import tensorflow as be
    return nn.Tensor(be.ones(shape, dtype='int32'), shape=shape)


def _create_variable(shape):
    return nn.get_variable(
        name='var', shape=shape, dtype='int32',
        initializer=nn.initializer.Constant(7)
    )


class TestTensorOps(unittest.TestCase):
    def test_mul_numbers(self):
        constant, shape = 10, (3, 5)
        with nn.variable_scope(self.id().replace('.', '/')):
            tensor1 = _create_ones_tensor(shape)
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
            tensor1 = _create_ones_tensor(shape)
            tensor2 = tensor1 * tensor1

        session = nn.Session()

        val1, val2 = session.run(
            outputs=[tensor1, tensor2],
        )
        self.assertTrue(np.all(val1 * val1 == val2))

    def test_mul_variable(self):
        shape = (3, 5)
        with nn.variable_scope(self.id().replace('.', '/')):
            tensor1 = _create_ones_tensor(shape)
            variable = _create_variable(shape)
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
            tensor1 = _create_ones_tensor(shape)
            input_ = nn.Input(name='input', shape=shape, dtype='int32')()
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
            tensor1 = _create_ones_tensor(shape)
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
            tensor1 = _create_ones_tensor(shape)
            tensor2 = tensor1 + tensor1

        session = nn.Session()
        val1, val2 = session.run(
            outputs=[tensor1, tensor2],
        )
        self.assertTrue(np.all(val1 + val1 == val2))

    def test_add_input(self):
        shape = (3, 5)
        with nn.variable_scope(self.id().replace('.', '/')):
            input_ = nn.Input(shape=shape, dtype='int32')()
            tensor1 = _create_ones_tensor(shape)
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
            tensor1 = _create_ones_tensor(shape)
            variable = _create_variable(shape)
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
            tensor1 = _create_ones_tensor(shape)
            tensor2 = -tensor1

        session = nn.Session()
        val1, val2 = session.run(
            outputs=[tensor1, tensor2],
        )
        self.assertTrue(np.all(-val1 == val2))

    def test_sub_numbers(self):
        constant, shape = 10, (3, 5)
        with nn.variable_scope(self.id().replace('.', '/')):
            tensor1 = _create_ones_tensor(shape)
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
            tensor1 = _create_ones_tensor(shape)
            tensor2 = tensor1 * tensor1

        session = nn.Session()

        val1, val2 = session.run(
            outputs=[tensor1, tensor2],
        )
        self.assertTrue(np.all(val1 * val1 == val2))

    def test_sub_input(self):
        shape = (3, 5)
        with nn.variable_scope(self.id().replace('.', '/')):
            input_ = nn.Input(shape=shape, dtype='int32')()
            tensor1 = _create_ones_tensor(shape)
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
            tensor1 = _create_ones_tensor(shape)
            variable = _create_variable(shape)
            tensor2 = tensor1 - variable
            tensor3 = variable - tensor1

        session = nn.Session()
        session.initialize()

        val1, val2, val3, val0 = session.run(
            outputs=[tensor1, tensor2, tensor3, variable],
        )
        self.assertTrue(np.all(val1 - val0 == val2))
        self.assertTrue(np.all(val0 - val1 == val3))
