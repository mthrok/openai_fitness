"""Unit test for luchador.nn.ops module"""
from __future__ import absolute_import

import numpy as np

import luchador
from luchador import nn
from tests.unit import fixture

_BACKEND = luchador.get_nn_backend()
# pylint: disable=invalid-name


###############################################################################
# Test gradients
class TestComputeGradiens(fixture.TestCase):
    """Test gradient computation"""
    def test_compute_gradients(self):
        """compute_gradients returns None for non-trainable wrt"""
        with nn.variable_scope(self.get_scope()):
            xs = [nn.make_variable(
                name='x_{}'.format(i), shape=(), trainable=bool(i % 2),
            ) for i in range(5)]
            y = xs[0] + xs[1] + xs[2] + xs[3] + xs[4]
            grads_and_vars = nn.ops.compute_gradient(loss=y, wrt=xs)
        self.assertEqual(len(xs), len(grads_and_vars))
        for i, (grad, var) in enumerate(grads_and_vars):
            self.assertIs(xs[i], var)
            if i % 2:
                self.assertIsNotNone(grad)
            else:
                self.assertIsNone(grad)

    def test_compute_gradients_with_trainables(self):
        """compute_gradients computes gradients for trainable wrt"""
        with nn.variable_scope(self.get_scope()):
            xs = [nn.make_variable(
                name='x_{}'.format(i), shape=(), trainable=True,
            ) for i in range(3)]
            y = xs[0] + xs[1] + xs[2]
            grads_and_vars = nn.ops.compute_gradient(loss=y, wrt=xs)
        self.assertEqual(len(xs), len(grads_and_vars))
        for i, (grad, var) in enumerate(grads_and_vars):
            self.assertIs(xs[i], var)
            self.assertIsNotNone(grad)

    def test_get_gradients(self):
        """gradients can be retrieved with get_tensor"""
        scope = self.get_scope()
        with nn.variable_scope(scope):
            xs = [nn.make_variable(
                name='x_{}'.format(i), shape=(), trainable=True,
            ) for i in range(5)]
            y = xs[0] + xs[1] + xs[2] + xs[3] + xs[4]
            grads_and_vars = nn.ops.compute_gradient(loss=y, wrt=xs)

            for i in range(5):
                grad = nn.get_tensor('{}_grad'.format(xs[i].name))
                self.assertIs(grads_and_vars[i][0], grad)

        for i in range(5):
            grad = nn.get_tensor('{}/{}_grad'.format(scope, xs[i].name))
            self.assertIs(grads_and_vars[i][0], grad)


###############################################################################
# Test clipping
class TestTensorOpsClipByValue(fixture.TestCase):
    """Test clipping wrapper by value"""
    def test_clip_number_by_value(self):
        """Test clip with float"""
        shape, min_value, max_value = (10, 10), 0.4, 0.6
        with nn.variable_scope(self.get_scope()):
            variable0 = fixture.create_random_variable(shape, dtype='float32')
            tensor1 = nn.ops.clip_by_value(
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
            tensor1 = nn.ops.clip_by_value(
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
            tensor1 = nn.ops.clip_by_value(
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
            tensor1 = nn.ops.clip_by_value(
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
    """Test clipping wrapper by norm"""
    def test_clip_number_by_norm(self):
        """Test clip_by_norm with float"""
        shape, clip_norm = (3, 4), 15.0
        with nn.variable_scope(self.get_scope()):
            input_ = nn.Input(shape, dtype='float32')
            output = nn.ops.clip_by_norm(input_, clip_norm=clip_norm)

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
            output = nn.ops.clip_by_norm(
                input_, clip_norm=clip_norm, axes=axis)

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
            output = nn.ops.clip_by_norm(input_, clip_norm=clip_var)

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


###############################################################################
# Test reduction operations
class _ReductionTest(fixture.TestCase):
    """Test case for reduce_** operations"""
    def _test(self, op, np_op, axis, shape, keep_dims):
        """Run reduce_** operation and check the result with NumPy func"""
        with nn.variable_scope(self.get_scope()):
            # pylint: disable=not-callable
            in_var = nn.Input(shape=shape)
            out_var = op(in_var, axis=axis, keep_dims=keep_dims)

        session = nn.Session()
        in_val = np.random.rand(*shape)
        out_val = session.run(
            outputs=out_var,
            inputs={in_var: in_val}
        )
        expected = np_op(in_val, axis=axis, keepdims=keep_dims)
        self.assertEqual(out_var.shape, out_val.shape)
        np.testing.assert_almost_equal(out_val, expected, decimal=3)


class TestReduceMean(_ReductionTest):
    """Test mean ops"""
    def _test_mean(self, axis, shape, keep_dims):
        self._test(nn.ops.reduce_mean, np.mean, axis, shape, keep_dims)

    def test_mean(self):
        """Test mean with single axis, dropping axis"""
        self._test_mean(0, (3, 5), False)

    def test_mean_keep_dim(self):
        """Test mean with single axis, keeping axis"""
        self._test_mean(0, (3, 5), True)

    def test_mean_multi(self):
        """Test mean with multiple axes, dropping axis"""
        self._test_mean((1, 2), (3, 4, 5, 6), False)

    def test_mean_multi_keep_dim(self):
        """Test mean with multiple axes, keeping axis"""
        self._test_mean((1, 2), (3, 4, 5, 6), True)

    def test_mean_no_axis(self):
        """Test mean with axis=Nones, dropping axis"""
        self._test_mean(None, (3, 4, 5, 6), False)

    def test_mean_no_axis_keep_dim(self):
        """Test mean with axis=Nones, keeping axis"""
        self._test_mean(None, (3, 4, 5, 6), True)


class TestReduceSum(_ReductionTest):
    """Test wrapper sum method"""
    def _test_sum(self, axis, shape, keep_dims):
        self._test(nn.ops.reduce_sum, np.sum, axis, shape, keep_dims)

    def test_sum(self):
        """Test sum with single axis, dropping axis"""
        self._test_sum(0, (3, 5), False)

    def test_sum_keep_dim(self):
        """Test sum with single axis, dropping axis"""
        self._test_sum(0, (3, 5), True)

    def test_sum_multi(self):
        """Test sum with multiple axes, dropping axis"""
        self._test_sum((1, 2), (3, 4, 5, 6), False)

    def test_sum_multi_keep_dim(self):
        """Test sum with multiple axes, dropping axis"""
        self._test_sum((1, 2), (3, 4, 5, 6), True)

    def test_sum_no_axis(self):
        """Test sum with axis=Nones, dropping axis"""
        self._test_sum(None, (3, 4, 5, 6), False)

    def test_sum_no_axis_keep_dim(self):
        """Test sum with axis=Nones, keeping axis"""
        self._test_sum(None, (3, 4, 5, 6), True)


class TestReduceMax(_ReductionTest):
    """Test wrapper max method"""
    def _test_max(self, axis, shape, keep_dims):
        self._test(nn.ops.reduce_max, np.max, axis, shape, keep_dims)

    def test_max(self):
        """Test max with single axis, dropping axis"""
        self._test_max(0, (3, 5), False)

    def test_max_keep_dim(self):
        """Test max with single axis, dropping axis"""
        self._test_max(0, (3, 5), True)

    def test_max_multi(self):
        """Test max with multiple axes, dropping axis"""
        self._test_max((1, 2), (3, 4, 5, 6), False)

    def test_max_multi_keep_dim(self):
        """Test max with multiple axes, dropping axis"""
        self._test_max((1, 2), (3, 4, 5, 6), True)

    def test_max_no_axis(self):
        """Test max with axis=Nones, dropping axis"""
        self._test_max(None, (3, 4, 5, 6), False)

    def test_max_no_axis_keep_dim(self):
        """Test max with axis=Nones, keeping axis"""
        self._test_max(None, (3, 4, 5, 6), True)


###############################################################################
# Test elementwise operations over multiple operands
class TestDot(fixture.TestCase):
    def _verify_shape(self, expected, found):
        for i, _ in enumerate(expected):
            if found[i] is None:
                continue
            self.assertEqual(found[i], expected[i])

    def _test_dot(self, shape0, shape1):
        with nn.variable_scope(self.get_scope()):
            in_var0 = nn.Input(shape=shape0)
            in_var1 = nn.Input(shape=shape1)
            out_var = nn.ops.dot(in_var0, in_var1)

        in_val0 = np.random.random(size=shape0)
        in_val1 = np.random.random(size=shape1)

        session = nn.Session()

        out_val = session.run(
            outputs=out_var,
            inputs={in_var0: in_val0, in_var1: in_val1},
        )
        np_val = np.dot(in_val0, in_val1)
        np.testing.assert_almost_equal(out_val, np_val, decimal=3)
        self._verify_shape(out_val.shape, out_var.shape)

    def test_dot_vector(self):
        """Test dot on vector"""
        self._test_dot((1, 4), (4, 1))
        self._test_dot((4, 1), (1, 4))

    def test_dot_2D(self):
        """Test dot 2D"""
        self._test_dot((3, 4), (4, 5))

    def test_dot_3D(self):
        """Test dot 3D"""
        self._test_dot((3, 4, 6), (5, 6, 7))
        self._test_dot((3, 4, 5), (5, 6))
        self._test_dot((3, 5), (4, 5, 6))

    def test_dot_4D(self):
        """Test dot 4D"""
        self._test_dot((3, 4, 5, 8), (6, 7, 8, 9))
        self._test_dot((3, 4, 5, 7), (6, 7, 8))
        self._test_dot((3, 4, 7), (5, 6, 7, 8))


###############################################################################
# Test elementwise operations over multiple operands
class _MultiElementwiseTest(fixture.TestCase):
    def _verify_shape(self, expected, found):
        for i, _ in enumerate(expected):
            if found[i] is None:
                continue
            self.assertEqual(found[i], expected[i])

    def _test_elementwise(self, shape0, shape1, op, np_op):
        with nn.variable_scope(self.get_scope()):
            in_var0 = nn.Input(shape=shape0, name='0')
            in_var1 = nn.Input(shape=shape1, name='1')
            out_var0 = op(in_var0, in_var1)
            out_var1 = op(in_var0, in_var1)
        session = nn.Session()

        shape0 = [32 if s is None else s for s in shape0]
        shape1 = [32 if s is None else s for s in shape1]

        in_val0 = np.random.uniform(size=shape0)
        in_val1 = np.random.uniform(size=shape1)
        out_val0, out_val1 = session.run(
            outputs=[out_var0, out_var1],
            inputs={in_var0: in_val0, in_var1: in_val1}
        )

        np.testing.assert_almost_equal(out_val0, np_op(in_val0, in_val1))
        np.testing.assert_almost_equal(out_val1, np_op(in_val1, in_val0))
        self._verify_shape(out_val0.shape, out_var0.shape)
        self._verify_shape(out_val1.shape, out_var1.shape)


class TestElementwiseMultiply(_MultiElementwiseTest):
    """Test mutiply operation"""
    def _test_multiply(self, shape0, shape1):
        self._test_elementwise(shape0, shape1, nn.ops.multiply, np.multiply)

    def test_same_shape(self):
        """Test multiply with the same shape"""
        shape = (3, 5)
        self._test_multiply(shape, shape)
        shape = (None, 3, 5)
        self._test_multiply(shape, shape)

    def test_same_dim_broadcast(self):
        """Test multiply with the same dimension"""
        self._test_multiply((3, 4, 5), (3, 1, 5))
        self._test_multiply((3, 1, 5), (3, 4, 5))
        self._test_multiply((3, 4, 5), (1, 4, 1))
        self._test_multiply((1, 4, 1), (3, 4, 5))
        self._test_multiply((1, 4, 5), (3, 1, 5))
        self._test_multiply((3, 4, 5), (3, 1, 1))
        self._test_multiply((None, 4, 8, 8), (32, 1, 8, 8))

        with self.assertRaises(ValueError):
            self._test_multiply((1, 4, 5), (3, 2, 5))

    def test_diff_dim_broadcast(self):
        """Test multiply with the different dimension"""
        self._test_multiply((3, 4, 5), (4, 5))
        self._test_multiply((4, 5), (3, 4, 5))
        self._test_multiply((3, 4, 5), (5,))
        self._test_multiply((5,), (3, 4, 5))
        self._test_multiply((3, 4, 5), (1,))
        self._test_multiply((1,), (3, 4, 5))

        with self.assertRaises(ValueError):
            self._test_multiply((3, 4, 5), (6, ))


class TestElementwiseAdd(_MultiElementwiseTest):
    """Test mutiply operation"""
    def _test_add(self, shape0, shape1):
        self._test_elementwise(shape0, shape1, nn.ops.add, np.add)

    def test_same_shape(self):
        """Test add with the same shape"""
        shape = (3, 5)
        self._test_add(shape, shape)
        shape = (None, 3, 5)
        self._test_add(shape, shape)

    def test_same_dim_broadcast(self):
        """Test add with the same dimension"""
        self._test_add((3, 4, 5), (3, 1, 5))
        self._test_add((3, 1, 5), (3, 4, 5))
        self._test_add((3, 4, 5), (1, 4, 1))
        self._test_add((1, 4, 1), (3, 4, 5))
        self._test_add((1, 4, 5), (3, 1, 5))
        self._test_add((3, 4, 5), (3, 1, 1))
        self._test_add((None, 4, 8, 8), (32, 1, 8, 8))

        with self.assertRaises(ValueError):
            self._test_add((1, 4, 5), (3, 2, 5))

    def test_diff_dim_broadcast(self):
        """Test add with the different dimension"""
        self._test_add((3, 4, 5), (4, 5))
        self._test_add((4, 5), (3, 4, 5))
        self._test_add((3, 4, 5), (5,))
        self._test_add((5,), (3, 4, 5))
        self._test_add((3, 4, 5), (1,))
        self._test_add((1,), (3, 4, 5))

        with self.assertRaises(ValueError):
            self._test_add((3, 4, 5), (6, ))


class TestElementwiseMaximum(_MultiElementwiseTest):
    """Test wrapper maximum method"""
    def _test_maximum(self, shape0, shape1):
        self._test_elementwise(shape0, shape1, nn.ops.maximum, np.maximum)

    def test_same_dim_broadcast(self):
        """Test maximum with the same dimension"""
        self._test_maximum((3, 4, 5), (3, 1, 5))
        self._test_maximum((3, 1, 5), (3, 4, 5))
        self._test_maximum((3, 4, 5), (1, 4, 1))
        self._test_maximum((1, 4, 1), (3, 4, 5))
        self._test_maximum((1, 4, 5), (3, 1, 5))
        self._test_maximum((3, 4, 5), (3, 1, 1))
        self._test_maximum((None, 4, 8, 8), (32, 1, 8, 8))

        with self.assertRaises(ValueError):
            self._test_maximum((1, 4, 5), (3, 2, 5))

    def test_diff_dim_broadcast(self):
        """Test maximum with the different dimension"""
        self._test_maximum((3, 4, 5), (4, 5))
        self._test_maximum((4, 5), (3, 4, 5))
        self._test_maximum((3, 4, 5), (5,))
        self._test_maximum((5,), (3, 4, 5))
        self._test_maximum((3, 4, 5), (1,))
        self._test_maximum((1,), (3, 4, 5))

        with self.assertRaises(ValueError):
            self._test_maximum((3, 4, 5), (6, ))


class TestElementwiseMinimum(_MultiElementwiseTest):
    """Test wrapper minimum method"""
    def _test_minimum(self, shape0, shape1):
        self._test_elementwise(shape0, shape1, nn.ops.minimum, np.minimum)

    def test_same_shape(self):
        """Test minimum with the same shape"""
        shape = (3, 5)
        self._test_minimum(shape, shape)
        shape = (None, 3, 5)
        self._test_minimum(shape, shape)

    def test_same_dim_broadcast(self):
        """Test minimum with the same dimension"""
        self._test_minimum((3, 4, 5), (3, 1, 5))
        self._test_minimum((3, 1, 5), (3, 4, 5))
        self._test_minimum((3, 4, 5), (1, 4, 1))
        self._test_minimum((1, 4, 1), (3, 4, 5))
        self._test_minimum((1, 4, 5), (3, 1, 5))
        self._test_minimum((3, 4, 5), (3, 1, 1))
        self._test_minimum((None, 4, 8, 8), (32, 1, 8, 8))

        with self.assertRaises(ValueError):
            self._test_minimum((1, 4, 5), (3, 2, 5))

    def test_diff_dim_broadcast(self):
        """Test minimum with the different dimension"""
        self._test_minimum((3, 4, 5), (4, 5))
        self._test_minimum((4, 5), (3, 4, 5))
        self._test_minimum((3, 4, 5), (5,))
        self._test_minimum((5,), (3, 4, 5))
        self._test_minimum((3, 4, 5), (1,))
        self._test_minimum((1,), (3, 4, 5))

        with self.assertRaises(ValueError):
            self._test_minimum((3, 4, 5), (6, ))


###############################################################################
# Test Misc
def _create_variables(shape=(3, 4)):
    init = nn.initializer.ConstantInitializer
    src = nn.make_variable('source', shape=shape, initializer=init(value=1))
    tgt = nn.make_variable('taget', shape=shape, initializer=init(value=0))
    return src, tgt


class TestOpsSync(fixture.TestCase):
    """Test ops.build_sync_op function"""
    def test_sync_without_tau(self):
        """sync op copies values from source variables to target variables"""
        with nn.variable_scope(self.get_scope()):
            source_var, target_var = _create_variables()
            sync_op = nn.ops.build_sync_op(
                [source_var], [target_var], tau=None)

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
        with nn.variable_scope(self.get_scope()):
            source_var, target_var = _create_variables()
            sync_op = nn.ops.build_sync_op(
                [source_var], [target_var], tau=tau)

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


class TestTensorOpsOneHot(fixture.TestCase):
    """Test wrapper one_hot"""
    def _test_one_hot(self, shape, n_classes, out_dtype):
        with nn.variable_scope(self.get_scope()):
            input_ = nn.Input(shape=shape, dtype='int64')
            tensor = nn.ops.one_hot(
                input_, n_classes=n_classes, dtype=out_dtype)

        session = nn.Session()

        in_val = np.random.randint(0, n_classes, size=shape)
        out_val = session.run(
            outputs=tensor,
            givens={
                input_: in_val,
            },
        )
        self.assertEqual(out_val.dtype, out_dtype)
        expected = np.zeros(shape=(shape[0], n_classes), dtype=out_dtype)
        expected[np.arange(shape[0]), in_val] = 1
        np.testing.assert_equal(out_val, expected)

    def test_one_hot_int32(self):
        """Test one hot conversion"""
        self._test_one_hot(shape=[10], n_classes=4, out_dtype='int32')

    def test_one_hot_int64(self):
        """Test one hot conversion"""
        self._test_one_hot(shape=[10], n_classes=4, out_dtype='int64')

    def test_one_hot_float32(self):
        """Test one hot conversion"""
        self._test_one_hot(shape=[10], n_classes=4, out_dtype='float32')

    def test_one_hot_float64(self):
        """Test one hot conversion"""
        self._test_one_hot(shape=[10], n_classes=4, out_dtype='float64')


###############################################################################
# Test Transform
class TestTensorOpsReshape(fixture.TestCase):
    """Test wrapper reshape method"""
    def _test_reshape(
            self, in_shape, out_shape, dtype='float32', in_val=None):
        """Test rehsape"""
        with nn.variable_scope(self.get_scope()):
            in_var = nn.Input(shape=in_shape, dtype=dtype)
            out_var = nn.ops.reshape(in_var, out_shape)

        session = nn.Session()

        if in_val is None:
            in_val = np.random.random(in_shape).astype(dtype)
        out_val = session.run(
            outputs=out_var,
            givens={
                in_var: in_val,
            },
        )
        expected = in_val.reshape(out_shape)
        np.testing.assert_equal(out_val, expected)
        if None not in in_shape:
            self.assertEqual(out_val.shape, out_var.shape)

    def test_reshape(self):
        """Test rehsape"""
        in_shape, out_shape = (3, 8), (1, 2, 2, 6)
        self._test_reshape(in_shape, out_shape)

    def test_reshape_minus_1(self):
        """Test rehsape with -1"""
        in_shape, out_shape = (3, 8), (1, 2, 2, -1)
        self._test_reshape(in_shape, out_shape)

    def test_reshape_with_none(self):
        """Test rehsape with None"""
        dtype = 'float32'
        in_shape, out_shape = (None, 8), (1, 2, 2, -1)
        in_val = np.random.random((3, 8)).astype(dtype)
        self._test_reshape(in_shape, out_shape, in_val=in_val, dtype=dtype)
        in_val = np.random.random((5, 8)).astype(dtype)
        self._test_reshape(in_shape, out_shape, in_val=in_val, dtype=dtype)


class TestOpsTile(fixture.TestCase):
    """Test wrapper tile method"""
    def _test_tile(self, in_shape, pattern, dtype='float32', in_val=None):
        """Test tile"""
        with nn.variable_scope(self.get_scope()):
            input_ = nn.Input(shape=in_shape, dtype=dtype)
            tensor = nn.ops.tile(input_, pattern=pattern)

        session = nn.Session()

        if in_val is None:
            in_val = np.random.random(in_shape).astype(dtype)
        out_val = session.run(
            outputs=tensor,
            givens={
                input_: in_val,
            },
        )
        self.assertEqual(out_val.dtype, dtype)
        expected = np.tile(in_val, pattern)
        np.testing.assert_equal(out_val, expected)

    def test_tile(self):
        """Test tile"""
        in_shape, pattern = (2, 3), (3, 4)
        self._test_tile(in_shape, pattern)

    def test_tile_none(self):
        """Test tile array with None in shape"""
        dtype, in_shape, pattern = 'float32', (None, 8), (3, 4)
        in_val = np.random.random((3, 8)).astype(dtype)
        self._test_tile(in_shape, pattern, in_val=in_val, dtype=dtype)
