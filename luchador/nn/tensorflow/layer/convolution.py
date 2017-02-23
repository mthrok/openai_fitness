"""Implement Convolution classes in Tensorflow"""
from __future__ import division
from __future__ import absolute_import

import logging
import warnings

import tensorflow as tf

import luchador
from ...base import layer as base_layer
from .. import scope, wrapper
from .common import get_initializers

__all__ = ['Conv2D']

_LG = logging.getLogger(__name__)


def _validate_padding(padding):
    msg = '`padding` must be either "SAME", "VALID", "full" or "half"'
    if not isinstance(padding, str):
        raise ValueError(msg)

    _padding = padding.lower()
    if _padding not in ['full', 'half', 'same', 'valid']:
        raise ValueError(msg)

    if _padding == 'full':
        msg = ('"full" is not supported in Tensorflow, '
               'and is replaced by "valid"')
        warnings.warn(msg)


def _validate_strides(strides):
    if isinstance(strides, int):
        return
    try:
        if (
                len(strides) in [2, 4] and
                all([isinstance(s, int) for s in strides])
        ):
            return
    except TypeError:
        pass
    raise ValueError(
        '`strides` must be either int, '
        'tuple of two ints, or tuple of four ints'
    )


def _map_padding(padding):
    if padding.upper() in ['HALF', 'SAME']:
        return 'SAME'
    else:
        return 'VALID'


def _check_filter_shape(
        input_shape, filter_shape, strides, data_format, padding):
    flt_h, flt_w = filter_shape[0], filter_shape[1]
    if data_format == 'NCHW':
        img_h, img_w = input_shape[2], input_shape[3]
        str_h, str_w = strides[2], strides[3]
    else:
        img_h, img_w = input_shape[1], input_shape[2]
        str_h, str_w = strides[1], strides[2]
    if padding == 'VALID':
        warn_w = bool((img_w - flt_w) % str_w)
        warn_h = bool((img_h - flt_h) % str_h)
    else:
        warn_w = bool((img_w - 1) % str_w)
        warn_h = bool((img_h - 1) % str_h)
    if warn_w:
        warnings.warn(
            'Convolution op will not cover the right side of the input.'
            'Check the width configuration of filter and stride.',
            RuntimeWarning
        )
    if warn_h:
        warnings.warn(
            'Convolution op will not cover the bottom part of the input.'
            'Check the height configuration of filter and stride.',
            RuntimeWarning
        )


class Conv2D(base_layer.BaseConv2D):
    """Implement Conv2D layer in Tensorflow.

    See :any:`BaseConv2D` for detail.
    """
    def _validate_args(self, padding, strides, **args):
        _validate_padding(padding)
        _validate_strides(strides)

    ###########################################################################
    def _get_format(self):
        return self.args.get('data_format', luchador.get_nn_conv_format())

    def _get_strides(self):
        st, fmt = self.args['strides'], self._get_format()
        if isinstance(st, int):
            st = (st, st)
        if len(st) == 2:
            if fmt == 'NCHW':
                st = (1, 1, st[0], st[1])
            else:
                st = (1, st[0], st[1], 1)
        return st

    def _get_weight_shape(self, input_shape):
        n_out, fmt = self.args['n_filters'], self._get_format()
        n_in = input_shape[1] if fmt == 'NCHW' else input_shape[3]
        height, width = self.args['filter_height'], self.args['filter_width']
        return (height, width, n_in, n_out)

    def _get_padding(self):
        return _map_padding(self.args['padding'])

    ###########################################################################
    def _instantiate_parameters(self, input_shape, input_dtype):
        _LG.debug('    Input: shape %s, dtype %s', input_shape, input_dtype)
        initializers = get_initializers(
            self.args.get('initializers') or {}, self.args['with_bias'])

        w_shape = self._get_weight_shape(input_shape)
        _check_filter_shape(
            input_shape, w_shape, self._get_strides(),
            self._get_format(), self._get_padding())
        weight = scope.get_variable(
            name='weight', shape=w_shape, dtype=input_dtype,
            initializer=initializers['weight'])
        self._add_parameter('weight', weight)

        if self.args['with_bias']:
            b_shape = (self.args['n_filters'],)
            bias = scope.get_variable(
                name='bias', shape=b_shape, dtype=input_dtype,
                initializer=initializers['bias'])
            self._add_parameter('bias', bias)

    def _build(self, input_tensor):
        if not self._parameter_variables:
            self._instantiate_parameters(
                input_tensor.shape, input_tensor.dtype)

        weight = self._get_parameter('weight').unwrap()
        strides = self._get_strides()
        name = self.args.get('name')
        cudnn = self.args.get('use_cudnn_on_gpu', True)
        fmt = self._get_format()
        padding = self._get_padding()
        output = tf.nn.conv2d(
            input_tensor.unwrap(), weight, strides=strides,
            padding=padding, use_cudnn_on_gpu=cudnn,
            data_format=fmt, name=name)

        if self.args['with_bias']:
            bias = self._get_parameter('bias').unwrap()
            output = tf.nn.bias_add(
                output, bias, data_format=fmt, name='output')
        return wrapper.Tensor(output, name='output')
