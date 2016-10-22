from __future__ import absolute_import

import unittest

import numpy as np

import luchador
from luchador.nn import (
    Input,
    Session,
    scope as scp,
    SSE2,
    SigmoidCrossEntropy,
)

'''
import logging
import theano
theano.config.optimizer = 'None'
theano.config.exception_verbosity = 'high'
logging.getLogger('luchador').setLevel(logging.DEBUG)
'''

BE = luchador.get_nn_backend()


class CostTest(unittest.TestCase):
    longMessage = True

    def _compute_cost(self, cost, target, logit):
        target_tensor = Input(shape=target.shape).build()
        logit_tensor = Input(shape=logit.shape).build()

        output_tensor = cost.build(target_tensor, logit_tensor)

        session = Session()
        output_value = session.run(
            outputs=output_tensor,
            inputs={
                logit_tensor: logit,
                target_tensor: target,
            },
        )
        return output_value

    def test_sce_element_wise(self):
        batch, n_classes = 32, 3

        shape = (batch, n_classes)
        label = np.random.randint(0, n_classes, batch)
        target = np.zeros(shape=shape)
        target[np.arange(batch), label] = 1
        logit = np.random.randn(*shape)

        with scp.variable_scope(self.id().replace('.', '/')):
            sce = SigmoidCrossEntropy(elementwise=True)
            sce_be = self._compute_cost(sce, target, logit)

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
        batch, n_classes = 32, 3

        shape = (batch, n_classes)
        label = np.random.randint(0, n_classes, batch)
        target = np.zeros(shape=shape)
        target[np.arange(batch), label] = 1
        logit = np.random.randn(*shape)

        with scp.variable_scope(self.id().replace('.', '/')):
            sce = SigmoidCrossEntropy(elementwise=False)
            sce_be = self._compute_cost(sce, target, logit)

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
        batch, n_classes = 32, 3

        shape = (batch, n_classes)
        target = np.random.randn(*shape)
        prediction = np.random.randn(*shape)

        with scp.variable_scope(self.id().replace('.', '/')):
            sse2 = SSE2(elementwise=True)
            sse2_be = self._compute_cost(sse2, target, prediction)

        sse2_np = np.square(target - prediction) / 2

        diff = np.abs(sse2_be - sse2_np)
        self.assertTrue(
            np.all(diff < 0.001),
            'SSE2 computed with NumPy and {be} is different: \n'
            'NumPy:\n{sce_np}\n{be}:\n{sce_be}'
            .format(be=BE, sce_be=sse2_be, sce_np=sse2_np)
        )

    def test_sse2_scalar(self):
        batch, n_classes = 32, 3

        shape = (batch, n_classes)
        target = np.random.randn(*shape)
        prediction = np.random.randn(*shape)

        with scp.variable_scope(self.id().replace('.', '/')):
            sse2 = SSE2(elementwise=False)
            sse2_be = self._compute_cost(sse2, target, prediction)

        sse2_np = np.square(target - prediction) / 2
        sse2_np = np.mean(np.sum(sse2_np, axis=1))

        diff = np.abs(sse2_be - sse2_np)
        self.assertTrue(
            np.all(diff < 0.001),
            'SSE2 computed with NumPy and {be} is different: \n'
            'NumPy:\n{sce_np}\n{be}:\n{sce_be}'
            .format(be=BE, sce_be=sse2_be, sce_np=sse2_np)
        )
