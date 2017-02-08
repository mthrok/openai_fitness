from __future__ import division
from __future__ import absolute_import

# import theano
# theano.config.optimizer = 'None'
# theano.config.exception_verbosity = 'high'

import luchador
from luchador import nn

from tests.unit import fixture

BE = luchador.get_nn_backend()

# pylint: disable=too-many-locals, invalid-name


def _get_y_equals_x_squared(scope, x_init):
    with nn.variable_scope(scope):
        x = nn.get_variable(
            name='x', shape=(), trainable=True,
            initializer=nn.initializer.Constant(x_init))
        y = x * x
    return x, y


def _get_slot_var(optimizer, slot_name, var_name=None):
    opt_name = optimizer.args['name']
    names = ['{}/{}'.format(opt_name, slot_name)]
    if var_name:
        names.append('{}/{}/{}'.format(opt_name, var_name, slot_name))

    for v in optimizer.get_parameter_variables():
        if v.name in names:
            return v
    raise ValueError('No slot was found')


class OptimizerGradientTest(fixture.TestCase):
    """Test gradient computation interface IO"""
    def test_compute_gradients_with_trainables(self):
        """compute_gradients computes gradients for trainable wrt"""
        with nn.variable_scope(self.get_scope()):
            xs = [nn.get_variable(
                name='x_{}'.format(i), shape=(), trainable=True,
            ) for i in range(3)]
            y = xs[0] + xs[1] + xs[2]

        sgd = nn.optimizer.SGD(learning_rate=0.01)
        grads_and_vars = sgd.compute_gradients(y, wrt=xs)
        self.assertEqual(len(xs), len(grads_and_vars))
        for i, (grad, var) in enumerate(grads_and_vars):
            self.assertIs(xs[i].unwrap(), var)
            self.assertIsNotNone(grad)

    def test_compute_gradients(self):
        """compute_gradients returns None for non-trainable wrt"""
        with nn.variable_scope(self.get_scope()):
            xs = [nn.get_variable(
                name='x_{}'.format(i), shape=(), trainable=bool(i % 2),
            ) for i in range(5)]
            y = xs[0] + xs[1] + xs[2] + xs[3] + xs[4]

        sgd = nn.optimizer.SGD(learning_rate=0.01)
        grads_and_vars = sgd.compute_gradients(y, wrt=xs)
        self.assertEqual(len(xs), len(grads_and_vars))
        for i, (grad, var) in enumerate(grads_and_vars):
            self.assertIs(xs[i].unwrap(), var)
            if i % 2:
                self.assertIsNotNone(grad)
            else:
                self.assertIsNone(grad)


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
        beta1_pow_tensor = _get_slot_var(adam, 'beta1_power')
        beta2_pow_tensor = _get_slot_var(adam, 'beta2_power')
        m_tensor = _get_slot_var(adam, 'm', var_name=x_tensor.name)
        v_tensor = _get_slot_var(adam, 'v', var_name=x_tensor.name)

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
        beta1_pow_tensor = _get_slot_var(adamax, 'beta1_power')
        m_tensor = _get_slot_var(adamax, 'm', var_name=x_tensor.name)
        u_tensor = _get_slot_var(adamax, 'u', var_name=x_tensor.name)

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
