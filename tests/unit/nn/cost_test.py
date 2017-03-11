"""Test nn.cost module"""
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


class CostUtilTest(unittest.TestCase):
    """Test cost-related utility functions"""
    def test_get_cost(self):
        """get_cost returns correct cost class"""
        for name, expected in get_all_costs().items():
            found = nn.fetch_cost(name)
            self.assertEqual(
                expected, found,
                'get_cost returned wrong cost Class. '
                'Expected: {}, Found: {}.'.format(expected, found)
            )


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
    return output_value, output_tensor


class CostSCETest(unittest.TestCase):
    """Test Sigmoid Cross Entropy test"""
    longMessage = True

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
            sce_be, sce_var = _compute_cost(sce, target, logit)

        x, z = logit, target
        sce_np = np.maximum(0, x) - x * z + np.log(1 + np.exp(-np.abs(x)))

        np.testing.assert_almost_equal(sce_be, sce_np, decimal=5)
        self.assertEqual(sce_be.shape, sce_var.shape)

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
            sce_be, sce_var = _compute_cost(sce, target, logit)

        x, z = logit, target
        sce_np = np.maximum(0, x) - x * z + np.log(1 + np.exp(-np.abs(x)))
        sce_np = np.mean(np.sum(sce_np, axis=1), axis=0)

        np.testing.assert_almost_equal(sce_be, sce_np, decimal=5)
        self.assertEqual(sce_be.shape, sce_var.shape)


class CostSSETest(unittest.TestCase):
    """Test Sum Squared Error test"""
    longMessage = True

    def test_sse_element_wise(self):
        """SSE output value is correct"""
        batch, n_classes = 32, 3

        shape = (batch, n_classes)
        target = np.random.randn(*shape)
        prediction = np.random.randn(*shape)

        with nn.variable_scope(self.id().replace('.', '/')):
            sse = luchador.nn.cost.SSE(elementwise=True)
            sse_be, sse_var = _compute_cost(sse, target, prediction)

        sse_np = np.square(target - prediction)

        np.testing.assert_almost_equal(sse_be, sse_np, decimal=5)
        self.assertEqual(sse_be.shape, sse_var.shape)

    def test_sse_scalar(self):
        """SSE output value is correct"""
        batch, n_classes = 32, 3

        shape = (batch, n_classes)
        target = np.random.randn(*shape)
        prediction = np.random.randn(*shape)

        with nn.variable_scope(self.id().replace('.', '/')):
            sse = luchador.nn.cost.SSE(elementwise=False)
            sse_be, sse_var = _compute_cost(sse, target, prediction)

        sse_np = np.square(target - prediction)
        sse_np = np.mean(np.sum(sse_np, axis=1))

        np.testing.assert_almost_equal(sse_be, sse_np, decimal=5)
        self.assertEqual(sse_be.shape, sse_var.shape)


class CostSoftmaxCrossEntropyTest(unittest.TestCase):
    """Test SofmaxCrossEntropy test"""
    longMessage = True

    def test_sce_element_wise(self):
        """SoftmaxCrossEntropy output value is correct"""
        batch, n_classes = 32, 3

        shape = (batch, n_classes)
        target = np.random.randn(*shape)
        prediction = np.random.randn(*shape)

        with nn.variable_scope(self.id().replace('.', '/')):
            sce = luchador.nn.cost.SoftmaxCrossEntropy(elementwise=True)
            sce_be, sce_var = _compute_cost(sce, target, prediction)

        xdev = prediction - np.max(prediction, 1, keepdims=True)
        log_sm = xdev - np.log(np.sum(np.exp(xdev), axis=1, keepdims=True))
        sce_np = - target * log_sm
        sce_np = np.sum(sce_np, axis=1)

        np.testing.assert_almost_equal(sce_be, sce_np, decimal=5)
        self.assertEqual(sce_be.shape, sce_var.shape)

    def test_sce_scalar(self):
        """SoftmaxCrossEntropy output value is correct"""
        batch, n_classes = 32, 3

        shape = (batch, n_classes)
        target = np.random.randn(*shape)
        prediction = np.random.randn(*shape)

        with nn.variable_scope(self.id().replace('.', '/')):
            sce = luchador.nn.cost.SoftmaxCrossEntropy(elementwise=False)
            sce_be, sce_var = _compute_cost(sce, target, prediction)

        xdev = prediction - np.max(prediction, 1, keepdims=True)
        log_sm = xdev - np.log(np.sum(np.exp(xdev), axis=1, keepdims=True))
        sce_np = - target * log_sm
        sce_np = np.mean(np.sum(sce_np, axis=1))

        np.testing.assert_almost_equal(sce_be, sce_np, decimal=5)
        self.assertEqual(sce_var.shape, sce_be.shape)
