from __future__ import absolute_import

import unittest

from tests.unit.fixture import get_optimizers
from luchador.nn import get_optimizer

OPTIMIZERS = get_optimizers()
N_OPTIMIZERS = 6

ARGS1 = {
    'SGD': {
        'learning_rate': 1e-4,
        'name': 'SGD',
    },
    'RMSProp': {
        'learning_rate': 1e-4,
        'decay': 0.95,
        'epsilon': 1e-10,
        'momentum': None,
        'name': 'RMSProp',
    },
    'NeonRMSProp': {
        'learning_rate': 1e-4,
        'decay': 0.95,
        'epsilon': 1e-10,
        'name': 'NeonRMSProp',
    },
    'GravesRMSProp': {
        'learning_rate': 1e-4,
        'decay1': 0.95,
        'decay2': 0.95,
        'epsilon': 1e-10,
        'name': 'GravesRMSProp',
    },
    'Adam': {
        'learning_rate': 0.01,
        'beta1': 0.99,
        'beta2': 0.99,
        'epsilon': 1e-6,
        'name': 'Adam',
    },
    'Adamax': {
        'learning_rate': 0.01,
        'beta1': 0.9,
        'beta2': 0.99,
        'epsilon': 1e-6,
        'name': 'Adamax',
    },
}

ARGS2 = {
    'SGD': {
        'learning_rate': 1e-3,
        'name': 'SGD',
    },
    'RMSProp': {
        'learning_rate': 1e-4,
        'decay': 0.95,
        'epsilon': 1e-10,
        'momentum': 0.5,
        'name': 'RMSProp',
    },
    'NeonRMSProp': {
        'learning_rate': 1e-3,
        'decay': 0.90,
        'epsilon': 1e-5,
        'name': 'NeonRMSProp',
    },
    'GravesRMSProp': {
        'learning_rate': 1e-3,
        'decay1': 0.90,
        'decay2': 0.90,
        'epsilon': 1e-5,
        'name': 'GravesRMSProp',
    },
    'Adam': {
        'learning_rate': 0.001,
        'beta1': 0.9,
        'beta2': 0.999,
        'epsilon': 1e-08,
        'name': 'Adam',
    },
    'Adamax': {
        'learning_rate': 0.01,
        'beta1': 0.9,
        'beta2': 0.3,
        'epsilon': 1e-6,
        'name': 'Adamax',
    },
}


def make_optimizers(args):
    return [Optimizer(**args[name]) for name, Optimizer in OPTIMIZERS.items()]


class OptimizerTest(unittest.TestCase):
    longMessage = True

    def test_optimizer_test_coverage(self):
        """All optimizers are tested"""
        self.assertEqual(
            len(OPTIMIZERS), N_OPTIMIZERS,
            'Number of optimizers are changed. (New optimizer is added?) '
            'Fix unittest to cover new optimizers'
        )

    def test_optimizer_equality(self):
        """Optimizers with same arguments must be equal"""
        optimizers1 = make_optimizers(ARGS1)
        optimizers2 = make_optimizers(ARGS1)
        for name, i1, i2 in zip(ARGS1, optimizers1, optimizers2):
            self.assertEqual(
                i1, i2,
                '{} optimizers constructed with the same arguments '
                'must equal each other.'.format(name))
            self.assertEqual(
                i2, i1,
                '{} optimizers constructed with the same arguments '
                'must equal each other.'.format(name))

    def test_optimizer_inequality(self):
        """Optimizers with different arguments must not be equal"""
        optimizers1 = make_optimizers(ARGS1)
        optimizers2 = make_optimizers(ARGS2)
        for name, i1, i2 in zip(ARGS1, optimizers1, optimizers2):
            self.assertNotEqual(
                i1, i2,
                '{} optimizers constructed with the differnet arguments '
                'must not equal each other.'.format(name))
        for name, i1, i2 in zip(ARGS2, optimizers1, optimizers2):
            self.assertNotEqual(
                i2, i1,
                '{} optimizers constructed with the differnet arguments '
                'must not equal each other.'.format(name))

    def test_optimizer_serialize(self):
        """`serialize` method should return constructor arguments """
        optimizers = make_optimizers(ARGS1)
        for optimizer in optimizers:
            args = optimizer.serialize()
            found = args['args']
            expected = ARGS1[args['name']]
            self.assertEqual(
                expected, found,
                '\nThe serialized arguments must '
                'equal to the constructor arguments')

    def test_optimizer_equality_serialize(self):
        """Optimizers recreated with serialize are identical to originals"""
        for optimizer0 in make_optimizers(ARGS1):
            args = optimizer0.serialize()
            optimizer1 = get_optimizer(args['name'])(**args['args'])
            expected = optimizer0
            found = optimizer1
            self.assertEqual(
                expected, found,
                '\nThe optimizer recreated from serialized arguments must '
                'equal to the original.'
            )
