from __future__ import absolute_import

import unittest

from tests.unit.fixture import get_all_initializers
from luchador.nn import get_initializer

INITIALIZERS = get_all_initializers()
N_INITIALIZERS = 5

ARGS1 = {
    'Constant': {
        'value': 100,
        'dtype': 'float32',
    },
    'Uniform': {
        'minval': 10,
        'maxval': 100,
        'seed': 1,
        'dtype': 'float32',
    },
    'Normal': {
        'mean': 0,
        'stddev': 1,
        'seed': 1,
        'dtype': 'float32',
    },
    'Xavier': {
        'uniform': True,
        'seed': 1,
        'dtype': 'float32',
    },
    'XavierConv2D': {
        'uniform': True,
        'seed': 1,
        'dtype': 'float32',
    }
}

ARGS2 = {
    'Constant': {
        'value': 10,
        'dtype': 'float64',
    },
    'Uniform': {
        'minval': 5,
        'maxval': 50,
        'seed': 10,
        'dtype': 'float64',
    },
    'Normal': {
        'mean': 1,
        'stddev': 10,
        'seed': 10,
        'dtype': 'float64',
    },
    'Xavier': {
        'uniform': False,
        'seed': 10,
        'dtype': 'float64',
    },
    'XavierConv2D': {
        'uniform': False,
        'seed': 10,
        'dtype': 'float64',
    }
}


def make_initializers(args):
    return [Initializer(**args[name])
            for name, Initializer in INITIALIZERS.items()]


class InitializerTest(unittest.TestCase):
    longMessage = True

    def test_initializer_test_coverage(self):
        """All initializers are tested"""
        self.assertEqual(
            len(INITIALIZERS), N_INITIALIZERS,
            'Number of initializers are changed. (New initializer is added?) '
            'Fix unittest to cover new initializers'
        )

    def test_get_initalizer(self):
        """get_initializer returns correct initalizer class"""
        for name, Initializer in INITIALIZERS.items():
            expected = Initializer
            found = get_initializer(name)
            self.assertEqual(
                expected, found,
                'get_initializer returned wrong initializer Class. '
                'Expected: {}, Found: {}.'.format(expected, found)
            )

    def test_initializer_equality(self):
        """Initializers with same arguments must be equal"""
        initializers1 = make_initializers(ARGS1)
        initializers2 = make_initializers(ARGS1)
        for name, i1, i2 in zip(ARGS1, initializers1, initializers2):
            self.assertEqual(
                i1, i2,
                '{} initializers constructed with the same arguments '
                'must equal each other.'.format(name))
            self.assertEqual(
                i2, i1,
                '{} initializers constructed with the same arguments '
                'must equal each other.'.format(name))

    def test_initializer_inequality(self):
        """Initializers with different arguments must not be equal"""
        initializers1 = make_initializers(ARGS1)
        initializers2 = make_initializers(ARGS2)
        for name, i1, i2 in zip(ARGS1, initializers1, initializers2):
            self.assertNotEqual(
                i1, i2,
                '{} initializers constructed with the differnet arguments '
                'must not equal each other.'.format(name))
        for name, i1, i2 in zip(ARGS2, initializers1, initializers2):
            self.assertNotEqual(
                i2, i1,
                '{} initializers constructed with the differnet arguments '
                'must not equal each other.'.format(name))

    def test_initializer_serialize(self):
        """`serialize` method should return constructor arguments """
        initializers = make_initializers(ARGS1)
        for initializer in initializers:
            args = initializer.serialize()
            found = args['args']
            expected = ARGS1[args['name']]
            self.assertEqual(
                expected, found,
                '\nThe serialized arguments must '
                'equal to the constructor arguments')

    def test_initializer_equality_serialize(self):
        """Initializers recreated with serialize are identical to originals"""
        for initializer0 in make_initializers(ARGS1):
            args = initializer0.serialize()
            initializer1 = get_initializer(args['name'])(**args['args'])
            expected = initializer0
            found = initializer1
            self.assertEqual(
                expected, found,
                '\nThe initializer recreated from serialized arguments must '
                'equal to the original.'
            )
