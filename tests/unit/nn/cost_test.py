from __future__ import absolute_import

import unittest

import numpy as np
# import theano
# theano.config.optimizer = 'None'
# theano.config.exception_verbosity = 'high'

import luchador
from luchador import nn
from tests.unit.fixture import get_all_costs


BE = luchador.get_nn_backend()


def _compute_cost(cost, target, logit):
    target_tensor = nn.Input(shape=target.shape)
    logit_tensor = nn.Input(shape=logit.shape)

    output_tensor = cost.build(target_tensor, logit_tensor)

    session = nn.Session()
    output_value = session.run(
        outputs=output_tensor,
        inputs={
            logit_tensor: logit,
            target_tensor: target,
        },
    )
    return output_value


class CostTest(unittest.TestCase):
    longMessage = True

    def test_get_cost(self):
        """get_cost returns correct cost class"""
        for name, expected in get_all_costs().items():
            found = nn.get_cost(name)
            self.assertEqual(
                expected, found,
                'get_cost returned wrong cost Class. '
                'Expected: {}, Found: {}.'.format(expected, found)
            )

    def test_sce_element_wise(self):
        """SigmoidCrossEntropy output value is correct"""
        batch, n_classes = 32, 3

        shape = (batch, n_classes)
        label = np.random.randint(0, n_classes, batch)
        target = np.zeros(shape=shape)
        target[np.arange(batch), label] = 1
        logit = np.random.randn(*shape)

        with nn.variable_scope(self.id().replace('.', '/')):
            sce = luchador.nn.cost.SigmoidCrossEntropy(elementwise=True)
            sce_be = _compute_cost(sce, target, logit)

        x, z = logit, target
        sce_np = np.maximum(0, x) - x * z + np.log(1 + np.exp(-np.abs(x)))

        diff = np.abs(sce_be - sce_np)
        self.assertTrue(
            np.all(diff < 0.001),
            'SCE computed with NumPy and {be} is different: \n'
            'NumPy:\n{sce_np}\n{be}:\n{sce_be}'
            .format(be=BE, sce_be=sce_be, sce_np=sce_np)
        )

    def test_sce_scalar(self):
        """SigmoidCrossEntropy output value is correct"""
        batch, n_classes = 32, 3

        shape = (batch, n_classes)
        label = np.random.randint(0, n_classes, batch)
        target = np.zeros(shape=shape)
        target[np.arange(batch), label] = 1
        logit = np.random.randn(*shape)

        with nn.variable_scope(self.id().replace('.', '/')):
            sce = luchador.nn.cost.SigmoidCrossEntropy(elementwise=False)
            sce_be = _compute_cost(sce, target, logit)

        x, z = logit, target
        sce_np = np.maximum(0, x) - x * z + np.log(1 + np.exp(-np.abs(x)))
        sce_np = np.mean(np.sum(sce_np, axis=1), axis=0)

        diff = abs(sce_be - sce_np)
        self.assertTrue(
            diff < 0.001,
            'SCE computed with NumPy and {be} is different: \n'
            'NumPy:\n{sce_np}\n{be}:\n{sce_be}'
            .format(be=BE, sce_be=sce_be, sce_np=sce_np)
        )

    def test_sse2_element_wise(self):
        """SSE2 output value is correct"""
        batch, n_classes = 32, 3

        shape = (batch, n_classes)
        target = np.random.randn(*shape)
        prediction = np.random.randn(*shape)

        with nn.variable_scope(self.id().replace('.', '/')):
            sse2 = luchador.nn.cost.SSE2(elementwise=True)
            sse2_be = _compute_cost(sse2, target, prediction)

        sse2_np = np.square(target - prediction) / 2

        diff = np.abs(sse2_be - sse2_np)
        self.assertTrue(
            np.all(diff < 0.001),
            'SSE2 computed with NumPy and {be} is different: \n'
            'NumPy:\n{sce_np}\n{be}:\n{sce_be}'
            .format(be=BE, sce_be=sse2_be, sce_np=sse2_np)
        )

    def test_sse2_element_wise_clip(self):
        """SSE2 output value is correct"""
        batch, n_classes = 32, 3
        max_delta, min_delta = 0.1, -0.1

        shape = (batch, n_classes)
        target = np.random.randn(*shape)
        prediction = np.random.randn(*shape)

        with nn.variable_scope(self.id().replace('.', '/')):
            sse2 = nn.cost.SSE2(
                max_delta=max_delta, min_delta=min_delta, elementwise=True)
            sse2_be = _compute_cost(sse2, target, prediction)

        delta = target - prediction
        delta = np.minimum(np.maximum(delta, min_delta), max_delta)
        sse2_np = np.square(delta) / 2

        diff = np.abs(sse2_be - sse2_np)
        self.assertTrue(
            np.all(diff < 0.001),
            'SSE2 computed with NumPy and {be} is different: \n'
            'NumPy:\n{sce_np}\n{be}:\n{sce_be}'
            .format(be=BE, sce_be=sse2_be, sce_np=sse2_np)
        )

    def test_sse2_scalar(self):
        """SSE2 output value is correct"""
        batch, n_classes = 32, 3

        shape = (batch, n_classes)
        target = np.random.randn(*shape)
        prediction = np.random.randn(*shape)

        with nn.variable_scope(self.id().replace('.', '/')):
            sse2 = luchador.nn.cost.SSE2(elementwise=False)
            sse2_be = _compute_cost(sse2, target, prediction)

        sse2_np = np.square(target - prediction) / 2
        sse2_np = np.mean(np.sum(sse2_np, axis=1))

        diff = np.abs(sse2_be - sse2_np)
        self.assertTrue(
            np.all(diff < 0.001),
            'SSE2 computed with NumPy and {be} is different: \n'
            'NumPy:\n{sce_np}\n{be}:\n{sce_be}'
            .format(be=BE, sce_be=sse2_be, sce_np=sse2_np)
        )
