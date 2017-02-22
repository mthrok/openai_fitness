from __future__ import absolute_import

import unittest

import numpy as np

from luchador import nn
from tests.unit import fixture


def _create_variables(shape=(3, 4)):
    init = nn.initializer.Constant
    with nn.variable_scope('source'):
        src = nn.get_variable('source', shape=shape, initializer=init(value=1))
    with nn.variable_scope('target'):
        tgt = nn.get_variable('taget', shape=shape, initializer=init(value=0))
    return src, tgt


class TestMisc(unittest.TestCase):
    def test_sync_without_tau(self):
        """sync op copies values from source variables to target variables"""
        with nn.variable_scope(self.id().replace('.', '/')):
            source_var, target_var = _create_variables()
            sync_op = nn.build_sync_op([source_var], [target_var], tau=None)

        session = nn.Session()
        session.initialize()

        src_val, tgt_val = session.run([source_var, target_var])
        self.assertTrue((src_val == 1).all())
        self.assertTrue((tgt_val == 0).all())

        session.run(updates=sync_op)

        src_val, tgt_val = session.run([source_var, target_var])
        self.assertTrue((src_val == 1).all())
        self.assertTrue((tgt_val == src_val).all())

    def test_sync_with_tau(self):
        """sync op copies weighted sum of source and target variables"""
        tau = 0.1
        with nn.variable_scope(self.id().replace('.', '/')):
            source_var, target_var = _create_variables()
            sync_op = nn.build_sync_op([source_var], [target_var], tau=tau)

        session = nn.Session()
        session.initialize()

        src_val, tgt_val = session.run([source_var, target_var])
        self.assertTrue((src_val == 1).all())
        self.assertTrue((tgt_val == 0).all())

        for _ in range(10):
            expected = tau * src_val + (1 - tau) * tgt_val
            session.run(updates=sync_op)
            src_val, found = session.run([source_var, target_var])
            self.assertTrue((src_val == 1).all())
            self.assertTrue(
                np.square(expected - found).sum() < 1e-10,
                '\nExpected: \n{}\nFound: \n{}'.format(expected, found)
            )
            tgt_val = found


class TestTensorOpsClipByValue(fixture.TestCase):
    """Test clipping wrapper by value"""
    def test_clip_number_by_value(self):
        """Test clip with float"""
        shape, min_value, max_value = (10, 10), 0.4, 0.6
        with nn.variable_scope(self.get_scope()):
            variable0 = fixture.create_random_variable(shape, dtype='float32')
            tensor1 = nn.clip_by_value(
                variable0, max_value=max_value, min_value=min_value)

        session = nn.Session()
        session.initialize()

        val0, val1 = session.run(
            outputs=[variable0, tensor1],
        )
        expected = np.clip(val0, a_max=max_value, a_min=min_value)
        np.testing.assert_almost_equal(val1, expected)

    def test_clip_variable_by_value(self):
        """Test clip with Variable"""
        shape, min_value, max_value = (10, 10), 0.4, 0.6
        with nn.variable_scope(self.get_scope()):
            variable0 = fixture.create_random_variable(shape, dtype='float32')
            min_variable = fixture.create_constant_variable(
                shape=[], dtype='float32', value=min_value, name='min_var')
            max_variable = fixture.create_constant_variable(
                shape=[], dtype='float32', value=max_value, name='max_var')
            tensor1 = nn.clip_by_value(
                variable0, max_value=max_variable, min_value=min_variable)

        session = nn.Session()
        session.initialize()

        val0, val1 = session.run(
            outputs=[variable0, tensor1],
        )
        expected = np.clip(val0, a_max=max_value, a_min=min_value)
        np.testing.assert_almost_equal(val1, expected)

    def test_clip_tensor_by_value(self):
        """Test clip with Tensor"""
        shape, min_value, max_value = (10, 10), 0.4, 0.6
        with nn.variable_scope(self.get_scope()):
            variable0 = fixture.create_random_variable(shape, dtype='float32')
            min_tensor = min_value * fixture.create_ones_tensor(
                shape=[], dtype='float32', name='min_tensor')
            max_tensor = max_value * fixture.create_ones_tensor(
                shape=[], dtype='float32', name='max_tensor')
            tensor1 = nn.clip_by_value(
                variable0, max_value=max_tensor, min_value=min_tensor)

        session = nn.Session()
        session.initialize()

        val0, val1 = session.run(
            outputs=[variable0, tensor1],
        )
        expected = np.clip(val0, a_max=max_value, a_min=min_value)
        np.testing.assert_almost_equal(val1, expected)

    def test_clip_input_by_value(self):
        """Test clip with Input"""
        shape, min_value, max_value = (10, 10), 0.4, 0.6
        with nn.variable_scope(self.get_scope()):
            variable0 = fixture.create_random_variable(shape, dtype='float32')
            min_input = nn.Input(shape=[], dtype='float32')
            max_input = nn.Input(shape=[], dtype='float32')
            tensor1 = nn.clip_by_value(
                variable0, max_value=max_input, min_value=min_input)

        session = nn.Session()
        session.initialize()

        val0, val1 = session.run(
            outputs=[variable0, tensor1],
            givens={
                min_input: np.array(min_value, dtype='float32'),
                max_input: np.array(max_value, dtype='float32'),
            },
        )
        expected = np.clip(val0, a_max=max_value, a_min=min_value)
        np.testing.assert_almost_equal(val1, expected)


class TestTensorOpsClipByNorm(fixture.TestCase):
    """Test clipping wrapper by vnorm"""
    def test_clip_number_by_norm(self):
        """Test clip_by_norm with float"""
        shape, clip_norm = (3, 4), 15.0
        with nn.variable_scope(self.get_scope()):
            input_ = nn.Input(shape, dtype='float32')
            output = nn.clip_by_norm(input_, clip_norm=clip_norm)

        session = nn.Session()

        in_val = np.random.rand(*shape).astype('float32')
        out_val = session.run(
            outputs=output,
            givens={input_: in_val}
        )
        np.testing.assert_almost_equal(out_val, in_val)

        in_val += 10.0
        out_val = session.run(
            outputs=output,
            givens={input_: in_val}
        )
        l2_norm = np.sqrt(np.sum(in_val ** 2))
        np.testing.assert_almost_equal(
            out_val, clip_norm * in_val / l2_norm, decimal=3)

    def test_clip_number_by_norm_with_axes(self):
        """Test clip_by_norm with axis"""
        shape, clip_norm, axis = (3, 4), 15.0, 1
        with nn.variable_scope(self.get_scope()):
            input_ = nn.Input(shape, dtype='float32')
            output = nn.clip_by_norm(input_, clip_norm=clip_norm, axes=axis)

        session = nn.Session()

        in_val = np.random.rand(*shape).astype('float32')
        out_val = session.run(
            outputs=output,
            givens={input_: in_val}
        )
        np.testing.assert_almost_equal(out_val, in_val)

        in_val += 10.0
        out_val = session.run(
            outputs=output,
            givens={input_: in_val}
        )
        l2_norm = np.sqrt(np.sum(in_val ** 2, axis=axis, keepdims=True))
        np.testing.assert_almost_equal(
            out_val, clip_norm * in_val / l2_norm, decimal=3)

    def test_clip_variable_by_norm(self):
        """Test clip_by_norm with Variable"""
        shape, clip_norm = (3, 4), np.asarray(15, dtype='float32')
        with nn.variable_scope(self.get_scope()):
            input_ = nn.Input(shape, dtype='float32')
            clip_var = nn.Input(shape=[], dtype='float32')
            output = nn.clip_by_norm(input_, clip_norm=clip_var)

        session = nn.Session()

        in_val = np.random.rand(*shape).astype('float32')
        out_val = session.run(
            outputs=output,
            givens={input_: in_val, clip_var: clip_norm}
        )
        np.testing.assert_almost_equal(out_val, in_val)

        in_val += 10.0
        out_val = session.run(
            outputs=output,
            givens={input_: in_val, clip_var: clip_norm}
        )
        l2_norm = np.sqrt(np.sum(in_val ** 2))
        np.testing.assert_almost_equal(
            out_val, clip_norm * in_val / l2_norm, decimal=3)
