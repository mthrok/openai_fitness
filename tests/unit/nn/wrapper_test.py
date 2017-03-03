"""Test wapper module"""
from __future__ import absolute_import

import numpy as np

from luchador import nn
from tests.unit import fixture

# pylint: disable=invalid-name


class TestTensorOpsMult(fixture.TestCase):
    """Test wrapper operations"""
    def test_mul_numbers(self):
        """Tensor * number is correct elementwise"""
        constant, shape = 10, (3, 5)
        with nn.variable_scope(self.get_scope()):
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
        with nn.variable_scope(self.get_scope()):
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
        with nn.variable_scope(self.get_scope()):
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
        with nn.variable_scope(self.get_scope()):
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


class TestTensorOpsDiv(fixture.TestCase):
    """Test wrapper division"""
    def test_truediv_numbers(self):
        """Tensor / number is correct elementwise"""
        constant, shape = 10., (3, 5)
        with nn.variable_scope(self.get_scope()):
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
        with nn.variable_scope(self.get_scope()):
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
        with nn.variable_scope(self.get_scope()):
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
        with nn.variable_scope(self.get_scope()):
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
        with nn.variable_scope(self.get_scope()):
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
        with nn.variable_scope(self.get_scope()):
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


class TestTensorOpsAdd(fixture.TestCase):
    """Test wrapper addition"""
    def test_add_numbers(self):
        """Tensor + number is correct elementwise"""
        constant, shape = 10, (3, 5)
        with nn.variable_scope(self.get_scope()):
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
        with nn.variable_scope(self.get_scope()):
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
        with nn.variable_scope(self.get_scope()):
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
        with nn.variable_scope(self.get_scope()):
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


class TestTensorOpsAbs(fixture.TestCase):
    """Test wrapper absolute op"""
    def test_abs(self):
        """abs(Tensor) is correct elementwise"""
        shape = (3, 5)
        with nn.variable_scope(self.get_scope()):
            input_ = nn.Input(shape, dtype='float64')
            output = abs(input_)

        input_val = np.random.randn(*shape)

        session = nn.Session()
        out_val = session.run(
            outputs=output,
            inputs={input_: input_val},
        )
        np.testing.assert_almost_equal(out_val, abs(input_val))


class TestTensorOpsNeg(fixture.TestCase):
    """Test wrapper negation"""
    def test_neg(self):
        """-Tensor is correct elementwise"""
        shape = (3, 5)
        with nn.variable_scope(self.get_scope()):
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
        with nn.variable_scope(self.get_scope()):
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
        with nn.variable_scope(self.get_scope()):
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
        with nn.variable_scope(self.get_scope()):
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
        with nn.variable_scope(self.get_scope()):
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


class TestGetInput(fixture.TestCase):
    """Test Input fetch"""
    def test_get_input(self):
        """Test if get_input correctly fetch Input object"""
        scope = self.get_scope()
        with nn.variable_scope(scope):
            input_0 = nn.Input(shape=[], name='input_a')
            input_1 = nn.get_input('input_a')

            self.assertIs(input_0, input_1)

            with self.assertRaises(ValueError):
                nn.get_input('input_b')

        input_2 = nn.get_input('{}/input_a'.format(scope))
        self.assertIs(input_0, input_2)

        with self.assertRaises(ValueError):
            nn.get_input('{}/input_b'.format(scope))
