"""Test Layer behaviors"""
from __future__ import division
from __future__ import absolute_import

import numpy as np

# import theano
# theano.config.optimizer = 'None'
# theano.config.exception_verbosity = 'high'

import luchador
from luchador import nn
from tests.unit.fixture import TestCase

_BE = luchador.get_nn_backend()
_FMT = luchador.get_nn_conv_format()


class Conv2DTransposeTest(TestCase):
    """Test for Conv2DTranspose class"""
    def _check(self, input_var, output_var):
        session = nn.Session()
        session.initialize()

        input_val = np.random.randn(*input_var.shape)
        output_val = session.run(
            outputs=output_var, inputs={input_var: input_val})

        self.assertEqual(output_var.shape, input_var.shape)
        self.assertEqual(output_var.dtype, input_var.dtype)
        self.assertEqual(output_var.shape, output_val.shape)
        self.assertEqual(output_var.dtype, output_val.dtype)

    def test_manual_construction(self):
        """Conv2DTranspose layer is built with provided output_shape"""
        h, w, c = 7, 5, 3
        strides, padding = 3, 'valid'
        if _FMT == 'NHWC' and _BE == 'tensorflow':
            input_shape = (32, 84, 84, 4)
        else:
            input_shape = (32, 4, 84, 84)

        conv2d = nn.layer.Conv2D(
            filter_height=h, filter_width=w, n_filters=c,
            strides=strides, padding=padding)
        input_var = nn.Input(shape=input_shape, name='original_input')

        with nn.variable_scope(self.get_scope('convolution')):
            conv_output = conv2d(input_var)

        conv2d_t = nn.layer.Conv2DTranspose(
            filter_height=h, filter_width=w, n_filters=c,
            strides=strides, padding=padding,
            output_shape=input_var.shape)

        with nn.variable_scope(self.get_scope('transpose')):
            conv_t_output = conv2d_t(conv_output)

        self._check(input_var, conv_t_output)
        self.assertIsNot(
            conv2d.get_parameter_variables('filter'),
            conv2d_t.get_parameter_variables('filter'),
        )

    def test_original_input(self):
        """Conv2DTranspose layer is built with provided original_input"""
        h, w, c = 7, 5, 3
        strides, padding = 3, 'valid'
        if _FMT == 'NHWC' and _BE == 'tensorflow':
            input_shape = (32, 84, 84, 4)
        else:
            input_shape = (32, 4, 84, 84)

        conv2d = nn.layer.Conv2D(
            filter_height=h, filter_width=w, n_filters=c,
            strides=strides, padding=padding)
        input_var = nn.Input(shape=input_shape, name='original_input')

        with nn.variable_scope(self.get_scope('convolution')):
            conv_output = conv2d(input_var)

        conv2d_t = nn.layer.Conv2DTranspose(
            filter_height=h, filter_width=w, n_filters=c,
            strides=strides, padding=padding)
        conv2d_t.set_parameter_variables(original_input=input_var)

        with nn.variable_scope(self.get_scope('transpose')):
            conv_t_output = conv2d_t(conv_output)

        self._check(input_var, conv_t_output)
        self.assertIsNot(
            conv2d.get_parameter_variables('filter'),
            conv2d_t.get_parameter_variables('filter'),
        )

    def test_original_filter(self):
        """Conv2DTranspose layer is built with provided original_filter"""
        h, w, c = 7, 5, 3
        strides, padding = 3, 'valid'
        if _FMT == 'NHWC' and _BE == 'tensorflow':
            input_shape = (32, 84, 84, 4)
        else:
            input_shape = (32, 4, 84, 84)

        conv2d = nn.layer.Conv2D(
            filter_height=h, filter_width=w, n_filters=c,
            strides=strides, padding=padding)

        input_var = nn.Input(shape=input_shape, name='original_input')

        with nn.variable_scope(self.get_scope('convolution')):
            conv_output = conv2d(input_var)

        original_filter = conv2d.get_parameter_variables('filter')
        conv2d_t = nn.layer.Conv2DTranspose(
            filter_height=h, filter_width=w, n_filters=c,
            strides=strides, padding=padding)
        conv2d_t.set_parameter_variables(
            original_input=input_var, original_filter=original_filter)

        with nn.variable_scope(self.get_scope('transpose')):
            conv_t_output = conv2d_t(conv_output)

        self._check(input_var, conv_t_output)
        self.assertIsNot(
            conv2d.get_parameter_variables('filter'),
            conv2d_t.get_parameter_variables('filter'),
        )

    def test_tied_weight(self):
        """Conv2DTranspose layer is built with provided filter parameter"""
        h, w, c = 7, 5, 3
        strides, padding = 3, 'valid'
        if _FMT == 'NHWC' and _BE == 'tensorflow':
            input_shape = (32, 84, 84, 4)
        else:
            input_shape = (32, 4, 84, 84)

        conv2d = nn.layer.Conv2D(
            filter_height=h, filter_width=w, n_filters=c,
            strides=strides, padding=padding)
        input_var = nn.Input(shape=input_shape, name='original_input')

        with nn.variable_scope(self.get_scope('convolution')):
            conv_output = conv2d(input_var)

        original_filter = conv2d.get_parameter_variables('filter')
        conv2d_t = nn.layer.Conv2DTranspose(
            filter_height=h, filter_width=w, n_filters=c,
            strides=strides, padding=padding)
        conv2d_t.set_parameter_variables(
            original_input=input_var, filter=original_filter)

        with nn.variable_scope(self.get_scope('transpose')):
            conv_t_output = conv2d_t(conv_output)

        self._check(input_var, conv_t_output)
        self.assertIs(
            conv2d.get_parameter_variables('filter'),
            conv2d_t.get_parameter_variables('filter'),
        )

    def test_output_shape_format_nchw(self):
        """`output_shape` is converted correctly"""
        h, w, c = 7, 5, 3
        strides, padding = 3, 'valid'
        input_shape_nhwc, input_shape_nchw = (32, 84, 84, 4), (32, 4, 84, 84)
        if _FMT == 'NHWC' and _BE == 'tensorflow':
            input_shape = input_shape_nhwc
        else:
            input_shape = input_shape_nchw

        conv2d = nn.layer.Conv2D(
            filter_height=h, filter_width=w, n_filters=c,
            strides=strides, padding=padding)
        input_var = nn.Input(shape=input_shape, name='original_input')

        with nn.variable_scope(self.get_scope('convolution')):
            conv_output = conv2d(input_var)

        conv2d_t = nn.layer.Conv2DTranspose(
            filter_height=h, filter_width=w, n_filters=c,
            strides=strides, padding=padding,
            output_shape=input_shape_nchw, output_shape_format='NCHW')

        with nn.variable_scope(self.get_scope('transpose')):
            conv_t_output = conv2d_t(conv_output)

        self._check(input_var, conv_t_output)
        self.assertIsNot(
            conv2d.get_parameter_variables('filter'),
            conv2d_t.get_parameter_variables('filter'),
        )

    def test_output_shape_format_nhwc(self):
        """`output_shape` is converted correctly"""
        h, w, c = 7, 5, 3
        strides, padding = 3, 'valid'
        input_shape_nhwc, input_shape_nchw = (32, 84, 84, 4), (32, 4, 84, 84)
        if _FMT == 'NHWC' and _BE == 'tensorflow':
            input_shape = input_shape_nhwc
        else:
            input_shape = input_shape_nchw

        conv2d = nn.layer.Conv2D(
            filter_height=h, filter_width=w, n_filters=c,
            strides=strides, padding=padding)
        input_var = nn.Input(shape=input_shape, name='original_input')

        with nn.variable_scope(self.get_scope('convolution')):
            conv_output = conv2d(input_var)

        conv2d_t = nn.layer.Conv2DTranspose(
            filter_height=h, filter_width=w, n_filters=c,
            strides=strides, padding=padding,
            output_shape=input_shape_nhwc, output_shape_format='NHWC')

        with nn.variable_scope(self.get_scope('transpose')):
            conv_t_output = conv2d_t(conv_output)

        self._check(input_var, conv_t_output)
        self.assertIsNot(
            conv2d.get_parameter_variables('filter'),
            conv2d_t.get_parameter_variables('filter'),
        )
