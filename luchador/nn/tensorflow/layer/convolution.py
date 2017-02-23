"""Implement Convolution classes in Tensorflow"""
from __future__ import division
from __future__ import absolute_import

import logging
import warnings

import tensorflow as tf

import luchador
from ...base import layer as base_layer
from ...base import getter
from .. import scope, wrapper

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
    return 'VALID'


def _get_initializers(config):
    """Get initializers for Conv2D

    Parameters
    ----------
    config : dict
        filter : dict
            Initializer configuration for ``filter`` parameter. If not present,
            :func:`luchador.nn.theano.initializer.Xavier` is used.
        bias : dict
            Initializer configuration for ``bias`` parameter. If not present,
            :func:`luchador.nn.theano.initializer.Constant` with
            ``value = 0.1`` is used.

    Returns
    -------
    dict
        Resulting initializers for ``filter`` and ``bias``
    """
    ret = {}

    cfg = config.get('filter', {'typename': 'Xavier'})
    type_ = cfg['typename']
    ret['filter'] = getter.get_initializer(type_)(**cfg.get('args', {}))

    cfg = config.get('bias', {'typename': 'Constant', 'args': {'value': 0.1}})
    type_ = cfg['typename']
    ret['bias'] = getter.get_initializer(type_)(**cfg.get('args', {}))
    return ret


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


def _get_strides(strides, data_format):
    if isinstance(strides, int):
        strides = (strides, strides)
    if len(strides) == 2:
        if data_format == 'NCHW':
            strides = (1, 1, strides[0], strides[1])
        else:
            strides = (1, strides[0], strides[1], 1)
    return strides


class Conv2D(base_layer.BaseConv2D):
    """Implement Conv2D layer in Tensorflow.

    See :any:`BaseConv2D` for detail.
    """
    # pylint: disable=redefined-builtin
    def _validate_args(self, padding, strides, **args):
        _validate_padding(padding)
        _validate_strides(strides)

    ###########################################################################
    def _get_format(self):
        return self.args.get('data_format', luchador.get_nn_conv_format())

    def _get_filter_shape(self, input_shape, data_format):
        n_in = input_shape[1] if data_format == 'NCHW' else input_shape[3]
        height, width = self.args['filter_height'], self.args['filter_width']
        return (height, width, n_in, self.args['n_filters'])

    ###########################################################################
    def _instantiate_parameters(self, w_shape, input_dtype):
        initializers = _get_initializers(self.args.get('initializers') or {})

        filter = scope.get_variable(
            name='filter', shape=w_shape, dtype=input_dtype,
            initializer=initializers['filter'])
        self._add_parameter('filter', filter)

        if self.args['with_bias']:
            b_shape = (self.args['n_filters'],)
            bias = scope.get_variable(
                name='bias', shape=b_shape, dtype=input_dtype,
                initializer=initializers['bias'])
            self._add_parameter('bias', bias)

    def _build(self, input_tensor):
        input_shape = input_tensor.shape
        data_format = self._get_format()
        filter_shape = self._get_filter_shape(input_shape, data_format)

        if not self._parameter_variables:
            self._instantiate_parameters(filter_shape, input_tensor.dtype)

        strides = _get_strides(self.args['strides'], data_format)
        padding = _map_padding(self.args['padding'])
        cudnn = self.args.get('use_cudnn_on_gpu', True)

        _check_filter_shape(
            input_shape, filter_shape, strides, data_format, padding)

        filter = self._get_parameter('filter')
        output = tf.nn.conv2d(
            input_tensor.unwrap(), filter.unwrap(),
            strides=strides, padding=padding, use_cudnn_on_gpu=cudnn,
            data_format=data_format, name=self.args.get('name'))

        if self.args['with_bias']:
            bias = self._get_parameter('bias').unwrap()
            output = tf.nn.bias_add(
                output, bias, data_format=data_format, name='output')
        return wrapper.Tensor(output, name='output')
