from __future__ import print_function

from luchador.common import get_subclasses
from luchador.nn import (
    scope,
    Tensor,
    Constant,
)


class Formula(object):
    pass


class x2(Formula):
    """ y = x2 """
    @staticmethod
    def get():
        x = scope.get_variable(name='x', shape=[], initializer=Constant(3))
        x_ = x.unwrap()
        y = Tensor(x_ * x_, shape=[])
        return {
            'loss': y,
            'wrt': x,
        }


class x6(Formula):
    """
    y = (x - 1.5) * (x - 1) * (x - 1) * (x + 1) * (x + 1) * (x + 1.5)
    https://www.google.com/search?q=y+%3D+(x-1.5)(x+-1)(x-1)(x%2B1)(x%2B1)(x%2B1.5)

    Global minimum: (x, y) = (0, -2.25)
    Local minimum: (x, y) = (+- 1.354, -0.29)
    """
    @staticmethod
    def get():
        x_ = scope.get_variable(name='x', shape=[], initializer=Constant(2.0))
        x = x_.unwrap()
        y = (x - 1.5) * (x - 1) * (x - 1) * (x + 1) * (x + 1) * (x + 1.5)
        y_ = Tensor(y, shape=[])
        return {
            'loss': y_,
            'wrt': x_,
        }


def print_formulae():
    print(' '.join([formula.__name__ for formula in get_subclasses(Formula)]))


if __name__ == '__main__':
    print_formulae()
