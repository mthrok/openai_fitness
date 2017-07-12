"""Test nn.cost module"""
from __future__ import division
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

    def _test_cost(
            self, cost, target, prediction, expected, elementwise, decimal=5):
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

        np.testing.assert_almost_equal(out_val, expected, decimal=decimal)
        self.assertEqual(out_val.shape, out_var.shape)

    def _test_cost_constant_target(
            self, cost, target, prediction, expected, elementwise, decimal=5):
        with nn.variable_scope(self.get_scope()):
            pred_var = nn.Input(shape=prediction.shape)
            out_var = cost(target, pred_var)

        session = nn.Session()
        out_val = session.run(
            outputs=out_var,
            inputs={
                pred_var: prediction,
            },
        )

        if not elementwise:
            expected = np.sum(np.mean(expected, axis=0))

        np.testing.assert_almost_equal(out_val, expected, decimal=decimal)
        self.assertEqual(out_val.shape, out_var.shape)


class SSETest(_CostTest):
    """Test Sum Squared Error"""
    def _test_sse(self, elementwise, batch=32, n_classes=3):
        target = np.random.randn(batch, n_classes)
        prediction = np.random.randn(batch, n_classes)
        expected = np.square(target - prediction)
        cost = luchador.nn.cost.SSE(elementwise=elementwise)
        self._test_cost(cost, target, prediction, expected, elementwise)

    def _test_sse_constant_target(self, elementwise, shape=(32, 3)):
        prediction = np.random.randn(*shape)
        cost = luchador.nn.cost.SSE(elementwise=elementwise)

        for target in np.arange(0.0, 1.0, 0.1):
            expected = np.square(target - prediction)
            self._test_cost_constant_target(
                cost, target, prediction, expected, elementwise)

    def test_sse_element_wise(self):
        """SSE output value is correct element-wise"""
        self._test_sse(elementwise=True)
        self._test_sse_constant_target(elementwise=True)

    def test_sse_scalar(self):
        """SSE output value is correct"""
        self._test_sse(elementwise=False)
        self._test_sse_constant_target(elementwise=False)


def _expected_sigmoid_ce(logit, target):
    x, z = logit, target
    return np.maximum(0, x) - x * z + np.log(1 + np.exp(-np.abs(x)))


class SigmoidCrossEntropyTest(_CostTest):
    """Test Sigmoid Cross Entropy"""
    def _test_sce(self, elementwise, batch=32, n_classes=3):
        label = np.random.randint(0, n_classes, batch)
        target = np.zeros(shape=(batch, n_classes))
        target[np.arange(batch), label] = 1
        logit = np.random.randn(batch, n_classes)
        expected = _expected_sigmoid_ce(logit, target)

        cost = luchador.nn.cost.SigmoidCrossEntropy(elementwise=elementwise)
        self._test_cost(cost, target, logit, expected, elementwise)

    def _test_sce_constant_target(self, elementwise, batch=32, n_classes=1):
        logit = np.random.randn(batch, n_classes)
        cost = luchador.nn.cost.SigmoidCrossEntropy(elementwise=elementwise)

        for target in np.arange(0.0, 1.0, 0.1):
            expected = _expected_sigmoid_ce(logit, target)
            self._test_cost_constant_target(
                cost, target, logit, expected, elementwise)

    def test_sigmoid_ce_elementwise(self):
        """SigmoidCrossEntropy output value is correct"""
        self._test_sce(elementwise=True)
        self._test_sce_constant_target(elementwise=True)

    def test_sigmoid_ce_scalar(self):
        """SigmoidCrossEntropy output value is correct"""
        self._test_sce(elementwise=False)
        self._test_sce_constant_target(elementwise=False)


def _normalize_rows(mat):
    row_sums = (mat ** 2).sum(axis=1)
    return mat / row_sums[:, np.newaxis]


def _expected_softmax_ce(logit, target):
    xdev = logit - np.max(logit, 1, keepdims=True)
    log_sm = xdev - np.log(np.sum(np.exp(xdev), axis=1, keepdims=True))
    return np.sum(- target * log_sm, axis=1)


class SoftmaxCrossEntropyTest(_CostTest):
    """Test SofmaxCrossEntropy"""
    def _test_sce(self, elementwise, batch=32, n_classes=3):
        target = _normalize_rows(np.random.randn(batch, n_classes))
        prediction = np.random.randn(batch, n_classes)
        expected = _expected_softmax_ce(prediction, target)
        cost = luchador.nn.cost.SoftmaxCrossEntropy(elementwise=elementwise)
        self._test_cost(cost, target, prediction, expected, elementwise)

    def _test_sce_constant_target(self, elementwise, batch=32, n_classes=3):
        target = _normalize_rows(np.random.randn(batch, n_classes))
        prediction = np.random.randn(batch, n_classes)
        expected = _expected_softmax_ce(prediction, target)
        cost = luchador.nn.cost.SoftmaxCrossEntropy(elementwise=elementwise)
        self._test_cost_constant_target(
            cost, target, prediction, expected, elementwise)

    def test_softmax_ce_elementwise(self):
        """SoftmaxCrossEntropy output value is correct"""
        self._test_sce(elementwise=True)
        self._test_sce_constant_target(elementwise=True)

    def test_softmax_ce_scalar(self):
        """SoftmaxCrossEntropy output value is correct"""
        self._test_sce(elementwise=False)
        self._test_sce_constant_target(elementwise=False)


def _kld(mean, stddev, clip_max=1e10, clip_min=1e-10):
    stddev2 = np.square(stddev)
    clipped = np.clip(stddev2, a_max=clip_max, a_min=clip_min)
    return (np.square(mean) + stddev2 - np.log(clipped) - 1) / 2


class NormalKLDivergenceTest(_CostTest):
    """Test NormalKMDivergence class"""
    def _test_kld(self, elementwise, clip_max=1e5, clip_min=1e-10):
        shape = (32, 3, 5)
        mean = np.random.randn(*shape)
        stddev = np.random.randn(*shape)
        expected = _kld(mean, stddev, clip_max=clip_max, clip_min=clip_min)
        cost = luchador.nn.cost.NormalKLDivergence(elementwise=elementwise)
        self._test_cost(cost, mean, stddev, expected, elementwise)

    def test_kld_element_wise(self):
        """NormalKLDivergence output value is correct"""
        self._test_kld(elementwise=True)

    def test_kld_scalar(self):
        """NormalKLDivergence output value is correct"""
        self._test_kld(elementwise=False)
