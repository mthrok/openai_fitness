from __future__ import division
from __future__ import absolute_import

# import theano
# theano.config.optimizer = 'None'
# theano.config.exception_verbosity = 'high'

import numpy as np

import luchador
from luchador import nn

from tests.unit import fixture

BE = luchador.get_nn_backend()
# pylint: disable=too-many-locals, invalid-name


def _get_y_equals_x_squared(scope, x_init):
    with nn.variable_scope(scope):
        x = nn.make_variable(
            name='x', shape=(), trainable=True,
            initializer=nn.initializer.ConstantInitializer(x_init))
        y = x * x
    return x, y


class OptimizerGradientTest(fixture.TestCase):
    """Test gradient computation interface IO"""
    def test_compute_gradients_with_trainables(self):
        """compute_gradients computes gradients for trainable wrt"""
        sgd = nn.optimizer.SGD(learning_rate=0.01)
        with nn.variable_scope(self.get_scope()):
            xs = [nn.make_variable(
                name='x_{}'.format(i), shape=(), trainable=True,
            ) for i in range(3)]
            y = xs[0] + xs[1] + xs[2]
            grads_and_vars = sgd.compute_gradients(y, wrt=xs)
        self.assertEqual(len(xs), len(grads_and_vars))
        for i, (grad, var) in enumerate(grads_and_vars):
            self.assertIs(xs[i], var)
            self.assertIsNotNone(grad)

    def test_compute_gradients(self):
        """compute_gradients returns None for non-trainable wrt"""
        sgd = nn.optimizer.SGD(learning_rate=0.01)
        with nn.variable_scope(self.get_scope()):
            xs = [nn.make_variable(
                name='x_{}'.format(i), shape=(), trainable=bool(i % 2),
            ) for i in range(5)]
            y = xs[0] + xs[1] + xs[2] + xs[3] + xs[4]
            grads_and_vars = sgd.compute_gradients(y, wrt=xs)
        self.assertEqual(len(xs), len(grads_and_vars))
        for i, (grad, var) in enumerate(grads_and_vars):
            self.assertIs(xs[i], var)
            if i % 2:
                self.assertIsNotNone(grad)
            else:
                self.assertIsNone(grad)

    def test_get_gradients(self):
        """gradients can be retrieved with get_tensor"""
        sgd = nn.optimizer.SGD(learning_rate=0.01)
        scope = self.get_scope()
        with nn.variable_scope(scope):
            xs = [nn.make_variable(
                name='x_{}'.format(i), shape=(), trainable=True,
            ) for i in range(5)]
            y = xs[0] + xs[1] + xs[2] + xs[3] + xs[4]
            grads_and_vars = sgd.compute_gradients(y, wrt=xs)

            for i in range(5):
                grad = nn.get_tensor('{}_grad'.format(xs[i].name))
                self.assertIs(grads_and_vars[i][0], grad)

        for i in range(5):
            grad = nn.get_tensor('{}/{}_grad'.format(scope, xs[i].name))
            self.assertIs(grads_and_vars[i][0], grad)

    def test_clip_gradients(self):
        """Gradients are clipped"""
        sgd = nn.optimizer.SGD(learning_rate=1.0)
        shape = (32, 1)
        with nn.variable_scope(self.get_scope()):
            initializer = nn.get_initializer(
                'UniformInitializer')(minval=-3, maxval=3)
            x = nn.make_variable(
                name='x', shape=shape, initializer=initializer)
            y = x * x / 2
            grads_and_vars = sgd.compute_gradients(nn.ops.reduce_sum(y), wrt=x)
            grads_and_vars = [
                (nn.ops.clip_by_value(grad, max_value=1, min_value=-1), var)
                for grad, var in grads_and_vars
            ]
            op = sgd.apply_gradients(grads_and_vars)

        session = nn.Session()
        session.initialize()

        val_0 = session.run(outputs=x)
        session.run(updates=op)
        val_1_be = session.run(outputs=x)

        val_1_np = np.zeros(shape)
        val_1_np[val_0 > 1] = val_0[val_0 > 1] - 1
        val_1_np[val_0 < -1] = val_0[val_0 < -1] + 1
        np.testing.assert_almost_equal(val_1_be, val_1_np)


class AdamTest(fixture.TestCase):
    """Test Adam Optimizer"""
    def test_beta_power_update(self):
        """Beta paramete is updated every time update is evaluated"""
        beta1, beta2, x_init_val, name = 0.9, 0.999, 3.0, 'Adam'

        adam = nn.optimizer.Adam(
            learning_rate=0.01, beta1=beta1, beta2=beta2, name=name)

        x_tensor, y_tensor = _get_y_equals_x_squared(
            scope=self.get_scope(), x_init=x_init_val)

        minimize_op = adam.minimize(loss=y_tensor, wrt=x_tensor)
        beta1_pow_tensor = adam.get_parameter_variable('beta1_power')
        beta2_pow_tensor = adam.get_parameter_variable('beta2_power')
        m_tensor = adam.get_parameter_variable('{}/m'.format(x_tensor.name))
        v_tensor = adam.get_parameter_variable('{}/v'.format(x_tensor.name))

        session = nn.Session()
        session.initialize()

        x_val_prev, m_val_prev, v_val_prev = x_init_val, 0, 0
        for i in range(1, 10):
            session.run(updates=minimize_op, name='optimization')

            beta1_pow_val, beta2_pow_val = session.run(
                outputs=[beta1_pow_tensor, beta2_pow_tensor],
                name='fetch1',
            )

            expected = beta1 ** (i + 1)
            found = beta1_pow_val
            diff = abs(expected - found)
            self.assertTrue(diff < 0.01,
                            'Beta1 is not correctly updated. '
                            'Expected: {}, Found: {}'.format(expected, found))

            expected = beta2 ** (i + 1)
            found = beta2_pow_val
            diff = abs(expected - found)
            self.assertTrue(diff < 0.01,
                            'Beta2 is not correctly updated. '
                            'Expected: {}, Found: {}'.format(expected, found))

            x_val = session.run(outputs=x_tensor, name='fetch2')
            self.assertTrue(
                0 <= x_val < x_val_prev,
                'The value of `x` must regress to zero at each update. '
                'Previous value: {}, current value: {}'
                .format(x_val_prev, x_val)
            )
            x_val_prev = x_val

            m_val, v_val = session.run(
                outputs=[m_tensor, v_tensor], name='fetch3')
            grad = 2 * x_val_prev
            expected = m_val_prev + (1.0 - beta1) * (grad - m_val_prev)
            found = m_val
            diff = abs(expected - found)
            self.assertTrue(
                diff < 0.01,
                'The value of `m` is not correctly updated. '
                'Expected: {}, Fround: {}'
                .format(expected, found)
            )
            m_val_prev = m_val

            expected = v_val_prev + (1.0 - beta2) * (grad*grad - v_val_prev)
            found = v_val
            diff = abs(expected - found)
            self.assertTrue(
                diff < 0.01,
                'The value of `v` is not correctly updated. '
                'Expected: {}, Fround: {}'
                .format(expected, found)
            )
            v_val_prev = v_val


class AdamaxTest(fixture.TestCase):
    def test_beta_power_update(self):
        """Beta parameter is updated every time update is evaluated"""
        beta1, beta2, x_init_val = 0.9, 0.999, 3.0
        adamax = nn.optimizer.Adamax(learning_rate=0.01, beta1=beta1)

        x_tensor, y_tensor = _get_y_equals_x_squared(
            scope=self.get_scope(), x_init=x_init_val)

        minimize_op = adamax.minimize(loss=y_tensor, wrt=x_tensor)
        beta1_pow_tensor = adamax.get_parameter_variable('beta1_power')
        m_tensor = adamax.get_parameter_variable('{}/m'.format(x_tensor.name))
        u_tensor = adamax.get_parameter_variable('{}/u'.format(x_tensor.name))

        session = nn.Session()
        session.initialize()

        x_val_prev, m_val_prev, u_val_prev = x_init_val, 0, 0
        for i in range(1, 10):
            session.run(updates=minimize_op, name='optimization')

            beta1_pow_val = session.run(
                outputs=beta1_pow_tensor, name='fetch1')

            expected = beta1 ** (i + 1)
            found = beta1_pow_val
            diff = abs(expected - found)
            self.assertTrue(diff < 0.01,
                            'Beta1 is not correctly updated. '
                            'Expected: {}, Found: {}'.format(expected, found))

            m_val, u_val = session.run(
                outputs=[m_tensor, u_tensor], name='fetch3')
            grad = 2 * x_val_prev
            expected = m_val_prev + (1.0 - beta1) * (grad - m_val_prev)
            found = m_val
            diff = abs(expected - found)
            self.assertTrue(
                diff < 0.01,
                'The value of `m` is not correctly updated. '
                'Expected: {}, Fround: {}'
                .format(expected, found)
            )
            m_val_prev = m_val

            expected = max(u_val_prev * beta2, abs(grad))
            found = u_val
            diff = abs(expected - found)
            self.assertTrue(
                diff < 0.01,
                'The value of `u` is not correctly updated. '
                'Expected: {}, Fround: {}'
                .format(expected, found)
            )
            u_val_prev = u_val

            x_val = session.run(outputs=x_tensor, name='fetch2')
            self.assertTrue(
                0 <= x_val < x_val_prev,
                'The value of `x` must regress to zero at each update. '
                'Previous value: {}, current value: {}'
                .format(x_val_prev, x_val)
            )
            x_val_prev = x_val
