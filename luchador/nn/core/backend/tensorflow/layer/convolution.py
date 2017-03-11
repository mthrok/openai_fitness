"""Implement Convolution classes in Tensorflow"""
from __future__ import division
from __future__ import absolute_import

import logging
import warnings

import tensorflow as tf

import luchador
from luchador.nn.core import common
from luchador.nn.core.base import fetch_initializer
from .. import wrapper

__all__ = ['Conv2D', 'Conv2DTranspose']
_LG = logging.getLogger(__name__)
# pylint: disable=no-member


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


def _get_filter_init(config):
    """Make filter initializer. Default to Xavier"""
    config = config or {'typename': 'XavierInitializer'}
    return fetch_initializer(
        config['typename'])(**config.get('args', {}))


def _get_bias_init(config):
    """Make bias initializer. Default to Constant (0.1)"""
    config = config or {
        'typename': 'ConstantInitializer', 'args': {'value': 0.1}}
    return fetch_initializer(
        config['typename'])(**config.get('args', {}))


def _get_format(data_format):
    return data_format or luchador.get_nn_conv_format()


class _Conv2DMixin(object):
    # pylint: disable=no-self-use, too-few-public-methods
    def _validate_args(self, padding, strides, **_):
        _validate_padding(padding)
        _validate_strides(strides)

    def _get_filter_shape(self, input_shape, data_format):
        n_in = input_shape[1] if data_format == 'NCHW' else input_shape[3]
        height, width = self.args['filter_height'], self.args['filter_width']
        return (height, width, n_in, self.args['n_filters'])

    def _build_filter(self, shape, dtype):
        init = _get_filter_init(self.args['initializers'].get('filter'))
        filter_ = wrapper.make_variable(
            name='filter', shape=shape, dtype=dtype, initializer=init)
        self.set_parameter_variables(filter=filter_)

    def _build_bias(self, shape, dtype):
        init = _get_bias_init(self.args['initializers'].get('bias'))
        bias = wrapper.make_variable(
            name='bias', shape=shape, dtype=dtype, initializer=init)
        self.set_parameter_variables(bias=bias)

    def _build_parameters(self, filter_shape, bias_shape, dtype):
        if self.get_parameter_variable('filter') is None:
            self._build_filter(shape=filter_shape, dtype=dtype)

        if not self.args['with_bias']:
            return

        if self.get_parameter_variable('bias') is None:
            self._build_bias(shape=bias_shape, dtype=dtype)


class Conv2D(_Conv2DMixin):
    """Implement Conv2D layer in Tensorflow.

    See :any:`BaseConv2D` for detail.
    """
    def _build(self, input_tensor):
        input_shape = input_tensor.shape
        data_format = _get_format(self.args.get('data_format'))
        filter_shape = self._get_filter_shape(input_shape, data_format)
        bias_shape = (filter_shape[3],)
        self._build_parameters(filter_shape, bias_shape, input_tensor.dtype)

        strides = _get_strides(self.args['strides'], data_format)
        padding = _map_padding(self.args['padding'])
        cudnn = self.args.get('use_cudnn_on_gpu', True)

        _check_filter_shape(
            input_shape, filter_shape, strides, data_format, padding)

        filter_ = self.get_parameter_variable('filter')
        output = tf.nn.conv2d(
            input_tensor.unwrap(), filter_.unwrap(),
            strides=strides, padding=padding, use_cudnn_on_gpu=cudnn,
            data_format=data_format)

        if self.args['with_bias']:
            bias = self.get_parameter_variable('bias').unwrap()
            output = tf.nn.bias_add(
                output, bias, data_format=data_format, name='output')
        return wrapper.Tensor(output, name='output')


class Conv2DTranspose(_Conv2DMixin):
    """Implement Conv2DTranspose layer in Theano.

    See :any:`BaseConv2DTranspose` for detail.
    """
    def _get_output_shape_from_arg(self):
        if not self.args.get('output_shape_format'):
            return self.args['output_shape']

        _be = luchador.get_nn_conv_format()
        if _be == self.args['output_shape_format']:
            return self.args['output_shape']

        if _be == 'NHWC':
            _LG.info('  * Converting `output_shape` to NHWC')
            return common.nchw2nhwc(self.args['output_shape'])
        _LG.info('  * Converting `output_shape` to NCHW')
        return common.nhwc2nchw(self.args['output_shape'])

    def _get_output_shape(self):
        if self.args['output_shape']:
            return self._get_output_shape_from_arg()
        elif self._parameter_variables['original_input'] is not None:
            return self._parameter_variables['original_input'].shape
        else:
            raise RuntimeError(
                'Output shape is not given. Output shape must be given as '
                'either constructor `ouptut_shape` parameter or parameter '
                'variable `original_input` via `set_parameter_variables`.'
            )

    def _get_filter_shape(self, output_shape, data_format):
        if self.get_parameter_variable('filter') is not None:
            return self.get_parameter_variable('filter').shape
        if self.get_parameter_variable('original_filter') is not None:
            return self.get_parameter_variable('original_filter').shape
        return super(Conv2DTranspose, self)._get_filter_shape(
            output_shape, data_format)

    def _build(self, input_tensor):
        output_shape = self._get_output_shape()

        if None in output_shape:
            raise ValueError('output shape must be fully known in TF backend.')

        data_format = _get_format(self.args.get('data_format'))
        filter_shape = self._get_filter_shape(output_shape, data_format)
        bias_shape = (filter_shape[2], )
        self._build_parameters(filter_shape, bias_shape, input_tensor.dtype)

        filter_ = self.get_parameter_variable('filter')
        padding = _map_padding(self.args['padding'])
        strides = _get_strides(self.args['strides'], data_format)
        tensor_ = tf.nn.conv2d_transpose(
            value=input_tensor.unwrap(), filter=filter_.unwrap(),
            output_shape=output_shape, padding=padding, strides=strides,
            data_format=data_format,
        )

        if self.args['with_bias']:
            bias = self.get_parameter_variable('bias')
            tensor_ = tf.nn.bias_add(
                tensor_, bias.unwrap(), data_format=data_format, name='output')
        return wrapper.Tensor(tensor_, name='output')
