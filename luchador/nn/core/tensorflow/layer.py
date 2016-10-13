from __future__ import division
from __future__ import absolute_import

import logging
import warnings

import tensorflow as tf

import luchador
from ..base import (
    get_layer,
    get_initializer,
    Layer as BaseLayer,
)
from .wrapper import (
    Tensor,
    Operation
)
from .scope import get_variable
from .initializer import (
    Constant,
    Xavier,
    XavierConv2D,
)


_LG = logging.getLogger(__name__)

__all__ = [
    'BaseLayer', 'get_layer',
    'Dense', 'Conv2D', 'ReLU', 'Flatten', 'TrueDiv',
]


def _wrap_output(tensor, name='output'):
    name = '{}/{}'.format(tf.get_variable_scope().name, name)
    return Tensor(tensor, name=name)


class TFLayer(BaseLayer):
    def get_update_operation(self):
        return Operation(tf.group(*self.update_operations.values()))


class Dense(TFLayer):
    def __init__(self, n_nodes, initializers={}):
        """Initialize dense layer.
        Activation function, such as ReLU is not included.
        Also called fully connected, affine, linear or inner product.

        Args:
          n_nodes (int): The number of internal neurons.
        """
        super(Dense, self).__init__(n_nodes=n_nodes, initializers=initializers)

    def _instantiate_initializers(self):
        init_cfg = self.args.get('initializers', {})
        if 'weight' not in self.initializers:
            cfg = init_cfg.get('weight')
            self.initializers['weight'] = (
                get_initializer(cfg['name'])(**cfg['args'])
                if cfg else Xavier()
            )
        if 'bias' not in self.initializers:
            cfg = init_cfg.get('bias')
            self.initializers['bias'] = (
                get_initializer(cfg['name'])(**cfg['args'])
                if cfg else Constant(0.1)
            )

    def _instantiate_parameter_variables(self, n_inputs):
        self._instantiate_initializers()

        dtype = luchador.get_nn_dtype()

        b_shape = (self.args['n_nodes'],)
        w_shape = (n_inputs, self.args['n_nodes'])

        b_init = self.initializers['bias'].unwrap()
        w_init = self.initializers['weight'].unwrap()

        self._add_parameter('weight', get_variable(
            name='weight', shape=w_shape, initializer=w_init, dtype=dtype))
        self._add_parameter('bias', get_variable(
            name='bias', shape=b_shape, initializer=b_init, dtype=dtype))

    def build(self, input_tensor):
        _LG.debug('    Building {}: {}'.format(type(self).__name__, self.args))
        if not self.parameter_variables:
            self._instantiate_parameter_variables(input_tensor.get_shape()[1])

        weight = self._get_parameter('weight')
        bias = self._get_parameter('bias')
        prod = tf.matmul(input_tensor.unwrap(), weight.unwrap())
        output = tf.add(prod, bias.unwrap(), name='output')
        return _wrap_output(output)


def _map_padding(padding):
    if padding.upper() in ['HALF', 'SAME']:
        return 'SAME'
    else:
        return 'VALID'


class Conv2D(TFLayer):
    """Apply convolution to input"""
    def __init__(self, filter_height, filter_width, n_filters, strides,
                 padding='VALID', initializers={}, **kwargs):
        """Initialize 2D convolution layer.
        Args:
          filter_height (int): filter height (== row)
          filter_weight (int): filter weight (== column)
          n_filters (int): #filters (== #output channels)
          strides (int, tuple of two int, or tuple of four int): stride
            - When given type is int, the output is subsampled by this factor
              in both width and height direction.
            - When given type is tuple of two int, the output is subsapmled
              `strides[0]` in height direction and `striders[1]` in width
              direction.
            - [Tensorflow only] When given type is tuple of four int, it must
              be consistent with the input data format. That is:
              - data_format=='NHWC' (default): [batch, height, width, channel]
              - data_format=='NCHW': [batch, channel, height, width]
          padding:
            - [tensorflow] (str): Either 'SAME' or 'VALID'
            - [theano] (str or int or tuple of two int): See Theano doc
          kwargs:
            - Tensorflow: Arguments passed to tf.nn.conv2d.
              'use_cudnn_on_gpu' and 'name'
        """
        super(Conv2D, self).__init__(
            filter_height=filter_height, filter_width=filter_width,
            n_filters=n_filters, strides=strides, padding=padding,
            initializers=initializers, **kwargs)

    def _validate_padding(self, padding):
        msg = '`padding` must be either "SAME", "VALID", "full" or "half"'
        if not isinstance(padding, str):
            raise ValueError(msg)

        _padding = padding.lower()
        if _padding not in ['full', 'half', 'same', 'valid']:
            raise ValueError(msg)

        if _padding == 'full':
            msg = ('"full" mode is not supported in tensorflow backend. '
                   'It will be replaced by "valid"')
            warnings.warn(msg)

    def _validate_strides(self, strides):
        if isinstance(strides, int):
            return
        try:
            if (
                    len(strides) in [2, 4] and
                    all(map(lambda s: isinstance(s, int), strides))
            ):
                return
        except Exception:
            pass
        raise ValueError(
            '`strides` must be either int, '
            'tuple of two ints, or tuple of four ints'
        )

    def _validate_args(self, args):
        self._validate_padding(args['padding'])
        self._validate_strides(args['strides'])

    ###########################################################################
    def _get_format(self):
        return self.args.get('data_format', luchador.get_nn_conv_format())

    def _get_strides(self):
        s, fmt = self.args['strides'], self._get_format()
        if isinstance(s, int):
            s = [s] * 2
        if len(s) == 2:
            s = (1, 1, s[0], s[1]) if fmt == 'NCHW' else (1, s[0], s[1], 1)
        return s

    def _get_weight_shape(self, input_shape):
        n_out, fmt = self.args['n_filters'], self._get_format()
        n_in = input_shape[1] if fmt == 'NCHW' else input_shape[3]
        height, width = self.args['filter_height'], self.args['filter_width']
        return (height, width, n_in, n_out)

    def _get_padding(self):
        return _map_padding(self.args['padding'])

    def _check_filter_shape(self, input_shape, filter_shape):
        flt_h, flt_w = filter_shape[0], filter_shape[1]
        strides = self._get_strides()
        if self._get_format() == 'NCHW':
            img_h, img_w = input_shape[2], input_shape[3]
            str_h, str_w = strides[2], strides[3]
        else:
            img_h, img_w = input_shape[1], input_shape[2]
            str_h, str_w = strides[1], strides[2]
        if self._get_padding() == 'VALID':
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

    def _instantiate_initializers(self):
        init_cfg = self.args.get('initializers', {})
        if 'weight' not in self.initializers:
            cfg = init_cfg.get('weight')
            self.initializers['weight'] = (
                get_initializer(cfg['name'])(**cfg['args'])
                if cfg else XavierConv2D()
            )
        if 'bias' not in self.initializers:
            cfg = init_cfg.get('bias')
            self.initializers['bias'] = (
                get_initializer(cfg['name'])(**cfg['args'])
                if cfg else Constant(0.1)
            )

    def _instantiate_parameter_variables(self, input_shape):
        _LG.debug('    Input shape: {}'.format(input_shape))
        self._instantiate_initializers()

        b_shape = (self.args['n_filters'],)
        w_shape = self._get_weight_shape(input_shape)

        self._check_filter_shape(input_shape, w_shape)

        b_init = self.initializers['bias'].unwrap()
        w_init = self.initializers['weight'].unwrap()

        dtype = luchador.get_nn_dtype()
        self._add_parameter('weight', get_variable(
            name='weight', shape=w_shape, initializer=w_init, dtype=dtype))
        self._add_parameter('bias', get_variable(
            name='bias', shape=b_shape, initializer=b_init, dtype=dtype))

    def build(self, input_tensor):
        _LG.debug('    Building {}: {}'.format(type(self).__name__, self.args))
        if not self.parameter_variables:
            self._instantiate_parameter_variables(input_tensor.get_shape())

        strides = self._get_strides()
        weight = self._get_parameter('weight')
        bias = self._get_parameter('bias')
        name = self.args.get('name')
        cudnn = self.args.get('use_cudnn_on_gpu', True)
        fmt = self._get_format()
        padding = self._get_padding()
        conv = tf.nn.conv2d(
            input_tensor.unwrap(), weight.unwrap(), strides=strides,
            padding=padding, use_cudnn_on_gpu=cudnn,
            data_format=fmt, name=name)
        output = tf.nn.bias_add(
            conv, bias.unwrap(), data_format=fmt, name='output')
        return _wrap_output(output)


class ReLU(TFLayer):
    """Applies Rectified Linear Unit"""
    def __init__(self):
        super(ReLU, self).__init__()

    def build(self, input_tensor):
        _LG.debug('    Building {}: {}'.format(type(self).__name__, self.args))
        output = tf.nn.relu(input_tensor.unwrap(), 'ouptut')
        return _wrap_output(output)


class Flatten(TFLayer):
    """Reshape batch into 2D (batch_size, n_features)"""
    def __init__(self):
        super(Flatten, self).__init__()

    def build(self, input_tensor):
        _LG.debug('    Building {}: {}'.format(type(self).__name__, self.args))
        in_shape = input_tensor.get_shape()
        n_nodes = reduce(lambda prod, dim: prod*dim, in_shape[1:], 1)
        out_shape = (-1, n_nodes)
        output = tf.reshape(input_tensor.unwrap(), out_shape, 'output')
        return _wrap_output(output)


class TrueDiv(TFLayer):
    """Applies element wise division"""
    def __init__(self, denom, dtype=None):
        super(TrueDiv, self).__init__(denom=denom, dtype=None)
        self.denom = None

    def _instantiate_denominator(self):
        dtype = self.args['dtype'] or luchador.get_nn_dtype()
        self.denom = tf.constant(
            self.args['denom'], dtype=dtype, name='denominator')

    def build(self, input_tensor):
        _LG.debug('    Building {}: {}'.format(type(self).__name__, self.args))
        if self.denom is None:
            self._instantiate_denominator()
        output = tf.truediv(input_tensor.unwrap(), self.denom, 'ouptut')
        return _wrap_output(output)
