"""Define/list up formulae to use in optimizer integration test"""
from __future__ import print_function

import abc

from luchador.util import fetch_subclasses
from luchador import nn

# pylint: disable=invalid-name


class _Formula(object):
    __metaclass__ = abc.ABCMeta

    @staticmethod
    def get():
        """Return loss nd wrt object"""
        pass


class x2(_Formula):
    """ y = x2 """
    @staticmethod
    def get():
        x = nn.make_variable(
            name='x', shape=[],
            initializer=nn.initializer.ConstantInitializer(3))
        y = x * x
        return {
            'loss': y,
            'wrt': x,
        }


class x6(_Formula):
    """
    y = (x - 1.5) * (x - 1) * (x - 1) * (x + 1) * (x + 1) * (x + 1.5)
    https://www.google.com/search?q=y+%3D+(x-1.5)(x+-1)(x-1)(x%2B1)(x%2B1)(x%2B1.5)

    Global minimum: (x, y) = (0, -2.25)
    Local minimum: (x, y) = (+- 1.354, -0.29)
    """
    @staticmethod
    def get():
        x = nn.make_variable(
            name='x', shape=[],
            initializer=nn.initializer.ConstantInitializer(2.0))
        y = (x - 1.5) * (x - 1) * (x - 1) * (x + 1) * (x + 1) * (x + 1.5)
        return {
            'loss': y,
            'wrt': x,
        }


def print_formulae():
    """List up avaialable formulae"""
    print(
        ' '.join([
            formula.__name__
            for formula in fetch_subclasses(_Formula)
        ])
    )


if __name__ == '__main__':
    print_formulae()
