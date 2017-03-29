"""Test nn.cost module"""
from __future__ import absolute_import

import numpy as np
# import theano
# theano.config.optimizer = 'None'
# theano.config.exception_verbosity = 'high'

import luchador
from luchador import nn
from tests.unit.fixture import TestCase, get_all_costs


class CostUtilTest(TestCase):
    """Test cost-related utility functions"""
    def test_fetch_cost(self):
        """fetch_cost returns correct cost class"""
        for name, expected in get_all_costs().items():
            found = nn.fetch_cost(name)
            self.assertEqual(
                expected, found,
                'get_cost returned wrong cost Class. '
                'Expected: {}, Found: {}.'.format(expected, found)
            )


class _CostTest(TestCase):
    longMessage = True

    def _test_cost(self, cost, target, prediction, expected, elementwise):
        with nn.variable_scope(self.get_scope()):
            target_var = nn.Input(shape=target.shape)
            pred_var = nn.Input(shape=prediction.shape)
            out_var = cost(target_var, pred_var)

        session = nn.Session()
        out_val = session.run(
            outputs=out_var,
            inputs={
                target_var: target,
                pred_var: prediction,
            },
        )

        if not elementwise:
            expected = np.sum(np.mean(expected, axis=0))

        np.testing.assert_almost_equal(out_val, expected, decimal=5)
        self.assertEqual(out_val.shape, out_var.shape)


class SSETest(_CostTest):
    """Test Sum Squared Error"""
    def _test_sse(self, elementwise):
        shape = (32, 3)
        target = np.random.randn(*shape)
        prediction = np.random.randn(*shape)

        expected = np.square(target - prediction)

        cost = luchador.nn.cost.SSE(elementwise=elementwise)
        self._test_cost(cost, target, prediction, expected, elementwise)

    def test_sse_element_wise(self):
        """SSE output value is correct"""
        self._test_sse(elementwise=True)

    def test_sse_scalar(self):
        """SSE output value is correct"""
        self._test_sse(elementwise=False)


class SigmoidCrossEntropyTest(_CostTest):
    """Test Sigmoid Cross Entropy"""
    def _test_sce(self, elementwise):
        batch, n_classes = 32, 3

        shape = (batch, n_classes)
        label = np.random.randint(0, n_classes, batch)
        target = np.zeros(shape=shape)
        target[np.arange(batch), label] = 1
        logit = np.random.randn(*shape)

        x, z = logit, target
        expected = np.maximum(0, x) - x * z + np.log(1 + np.exp(-np.abs(x)))

        cost = luchador.nn.cost.SigmoidCrossEntropy(elementwise=elementwise)
        self._test_cost(cost, target, logit, expected, elementwise)

    def test_sce_elementwise(self):
        """SigmoidCrossEntropy output value is correct"""
        self._test_sce(elementwise=True)

    def test_sce_scalar(self):
        """SigmoidCrossEntropy output value is correct"""
        self._test_sce(elementwise=False)


class SoftmaxCrossEntropyTest(_CostTest):
    """Test SofmaxCrossEntropy"""
    def _test_sce(self, elementwise):
        shape = (32, 3)
        target = np.random.randn(*shape)
        prediction = np.random.randn(*shape)

        xdev = prediction - np.max(prediction, 1, keepdims=True)
        log_sm = xdev - np.log(np.sum(np.exp(xdev), axis=1, keepdims=True))
        expected = - target * log_sm
        expected = np.sum(expected, axis=1)

        cost = luchador.nn.cost.SoftmaxCrossEntropy(elementwise=elementwise)
        self._test_cost(cost, target, prediction, expected, elementwise)

    def test_sce_element_wise(self):
        """SoftmaxCrossEntropy output value is correct"""
        self._test_sce(elementwise=True)

    def test_sce_scalar(self):
        """SoftmaxCrossEntropy output value is correct"""
        self._test_sce(elementwise=False)
