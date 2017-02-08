from __future__ import absolute_import

import numpy as np

import luchador
from luchador import nn
from tests.unit import fixture


class SessionTest(fixture.TestCase):
    def _test_load_dataset(self, dtype1, dtype2):
        name = 'test_load_dataset_{}_{}'.format(dtype1, dtype2)
        shape = (3, 3)
        target_value = 10

        variable = nn.get_variable(name=name, shape=shape, dtype=dtype1)
        value = target_value * np.ones(shape, dtype=dtype2)

        session = nn.Session()
        session.load_dataset({name: value}, cast=not dtype1 == dtype2)

        updated_value = session.run(outputs=variable)
        self.assertTrue(np.all(target_value == updated_value))

    def test_load_dataset_float32(self):
        """Variable value is set via Session.load_dataset when dtype=float32"""
        self._test_load_dataset('float32', 'float32')

    def test_load_dataset_float64(self):
        """Variable value is set via Session.load_dataset when dtype=float64"""
        self._test_load_dataset('float64', 'float64')

    def test_load_dataset_downcast(self):
        """Variable value is set when data is downcasted"""
        self._test_load_dataset('float32', 'float64')

    def test_load_dataset_upcast(self):
        """Variable value is set when data is downcasted"""
        self._test_load_dataset('float64', 'float32')

    def test_cache_outputs(self):
        """Session.run works without giving output after tagging"""
        with nn.variable_scope(self.get_scope()):
            x = nn.Input(shape=(), name='x')
            y = x * x
            session = nn.Session()

            with self.assertRaises(Exception):
                session.run(outputs=y)

            val0 = 3.

            val1 = session.run(name='test_cache', inputs={x: val0}, outputs=y)
            val2 = session.run(name='test_cache', inputs={x: val0})

            np.testing.assert_almost_equal(val1, val0 * val0)
            np.testing.assert_almost_equal(val2, val0 * val0)

    def test_cache_updates(self):
        """Session.run works without giving updates after tagging"""
        w_0 = 6
        with nn.variable_scope(self.get_scope()):
            x = nn.Input(shape=(), name='x')
            initializer = nn.initializer.Constant(w_0)
            w = nn.get_variable(
                name='w', shape=(),
                initializer=initializer
            )
            y = w * x

            sgd = nn.optimizer.SGD(learning_rate=1.0)
            update_op = sgd.minimize(y, w)

            session = nn.Session()
            session.initialize()

            with self.assertRaises(Exception):
                session.run(updates=update_op)

            val0 = 3.

            val1 = session.run(name='test_cache', inputs={x: val0},
                               outputs=w, updates=update_op)
            val2 = session.run(name='test_cache', inputs={x: val0})

            # Tensorflow updates variables after evaluating output variables
            # Theano does not necessarily do the same, and we do not have controll
            # over it.
            if luchador.get_nn_backend() == 'tensorflow':
                np.testing.assert_almost_equal(val1, w_0 - val0)
                np.testing.assert_almost_equal(val2, w_0 - val0 - val0)
            else:
                np.testing.assert_almost_equal(val1, w_0)
                np.testing.assert_almost_equal(val2, w_0 - val0)
