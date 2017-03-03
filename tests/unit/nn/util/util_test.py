"""Test nn.util module"""
from __future__ import absolute_import

from luchador import nn
from tests.unit import fixture


class UtilTest(fixture.TestCase):
    """Test utility functions"""
    def test_apply_gradient_directory(self):
        """get_grad correctly fetches gradient Tensor from Variable"""
        w_0 = 6
        sgd = nn.optimizer.SGD(learning_rate=1.0)
        with nn.variable_scope(self.get_scope()):
            x = nn.Input(shape=(), name='x')
            w1 = nn.make_variable(
                name='w', shape=(),
                initializer=nn.initializer.ConstantInitializer(w_0),
            )
            y1 = w1 * x
            sgd.minimize(y1, w1)
            dy1dw1_1 = nn.get_tensor('{}_grad'.format(w1.name))
            dy1dw1_2 = nn.get_grad(w1)

            self.assertIs(dy1dw1_1, dy1dw1_2)

        with nn.variable_scope('{}/2'.format(self.get_scope())):
            w2 = nn.make_variable(
                name='w', shape=(),
                initializer=nn.initializer.ConstantInitializer(w_0),
            )
            y2 = w2 * x
            sgd.minimize(y2, w2)
            dy2dw2_1 = nn.get_tensor('{}_grad'.format(w2.name))
            dy2dw2_2 = nn.get_grad(w2)

            self.assertIs(dy2dw2_1, dy2dw2_2)
