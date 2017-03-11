from __future__ import absolute_import

import numpy as np

from luchador import nn
from tests.unit import fixture


class SessionTest(fixture.TestCase):
    def _test_load_dataset(self, dtype1, dtype2):
        name = 'test_load_dataset_{}_{}'.format(dtype1, dtype2)
        shape = (3, 3)
        target_value = 10

        variable = nn.make_variable(name=name, shape=shape, dtype=dtype1)
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

    def test_apply_gradient_directory(self):
        """Variables can be updated by appyling gradient directly"""
        w_0 = 6
        with nn.variable_scope(self.get_scope()):
            x = nn.Input(shape=(), name='x')
            w = nn.make_variable(
                name='w', shape=(),
                initializer=nn.initializer.ConstantInitializer(w_0),
            )
            y = w * x

            sgd = nn.optimizer.SGD(learning_rate=1.0)
            update_op = sgd.minimize(y, w)
            dw = nn.get_tensor('{}_grad'.format(w.name))

            session = nn.Session()
            session.initialize()

            val0 = 3.
            session.run(updates=update_op, givens={dw: val0})
            val_w = session.run(outputs=w)

            np.testing.assert_almost_equal(val_w, w_0 - val0)

    def test_check_optimizer_slot(self):
        """Slot variables are updated when applying gradient directly"""
        name, b1_0, b2_0 = 'Adam', 0.5, 0.4
        opt = nn.optimizer.Adam(
            learning_rate=1.0, name=name, beta1=b1_0, beta2=b2_0)
        with nn.variable_scope(self.get_scope()):
            x = nn.Input(shape=(), name='x')
            w = nn.make_variable(shape=(), name='w')
            update_op = opt.minimize(w * x, w)

            dw = nn.get_tensor('{}_grad'.format(w.name))
            b1 = nn.get_variable('{}/beta1_power'.format(name))
            b2 = nn.get_variable('{}/beta2_power'.format(name))

        session = nn.Session()
        session.initialize()

        for i in range(10):
            b1_val, b2_val = session.run(outputs=[b1, b2])
            np.testing.assert_almost_equal(b1_val, b1_0 ** (i + 1))
            np.testing.assert_almost_equal(b2_val, b2_0 ** (i + 1))
            session.run(updates=update_op, givens={dw: 1.0})
