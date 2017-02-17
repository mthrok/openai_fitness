"""Implement Layer classes in Tensorflow"""
from __future__ import division
from __future__ import absolute_import

import logging
import warnings

import tensorflow as tf

import luchador
from ..base import (
    getter,
    layer as base_layer,
)
from . import scope, wrapper, initializer

# pylint: disable=too-few-public-methods, invalid-name

__all__ = [
    'LayerMixin',
    'Dense', 'Conv2D',
    'ReLU', 'Softplus',
    'Sigmoid', 'Softmax',
    'Tanh', 'Sin', 'Cos',
    'Flatten', 'TrueDiv', 'Mean',
    'Concat', 'Add', 'Sub',
    'BatchNormalization',
    'NHWC2NCHW', 'NCHW2NHWC',
]

_LG = logging.getLogger(__name__)


class LayerMixin(object):
    """Implement the following common Layer methods in Tensorflow

    - ``_get_update_operation``

    """
    def _get_update_operation(self):
        return wrapper.Operation(tf.group(*self.update_operations.values()))


def _get_initializers(cfg, with_bias):
    """Initializer for Dense and Conv2D"""
    w_cfg = cfg.get('weight')
    ret = {}
    ret['weight'] = (
        getter.get_initializer(w_cfg['typename'])(**w_cfg['args'])
        if w_cfg else initializer.Xavier()
    )

    if with_bias:
        _cfg = cfg.get('bias')
        ret['bias'] = (
            getter.get_initializer(_cfg['typename'])(**_cfg['args'])
            if _cfg else initializer.Constant(0.1)
        )

    return ret


class Dense(LayerMixin, base_layer.BaseDense):
    """Implement Dense layer in Tensorflow.

    See :any:`BaseDense` for detail.
    """
    def _instantiate_parameters(self, n_inputs, dtype):
        initializers = _get_initializers(
            self.args.get('initializers') or {}, self.args['with_bias'])

        w_shape = (n_inputs, self.args['n_nodes'])
        weight = scope.get_variable(
            name='weight', shape=w_shape, dtype=dtype,
            initializer=initializers['weight'])
        self._add_parameter('weight', weight)

        if self.args['with_bias']:
            b_shape = (self.args['n_nodes'],)
            bias = scope.get_variable(
                name='bias', shape=b_shape, dtype=dtype,
                initializer=initializers['bias'])
            self._add_parameter('bias', bias)

    def _build(self, input_tensor):
        if not self._parameter_variables:
            self._instantiate_parameters(
                input_tensor.shape[1], input_tensor.dtype)

        weight = self._get_parameter('weight').unwrap()
        output = tf.matmul(input_tensor.unwrap(), weight)

        if self.args['with_bias']:
            bias = self._get_parameter('bias').unwrap()
            output = tf.add(output, bias, name='output')
        return wrapper.Tensor(output, name='output')


def _map_padding(padding):
    if padding.upper() in ['HALF', 'SAME']:
        return 'SAME'
    else:
        return 'VALID'


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


class Conv2D(LayerMixin, base_layer.BaseConv2D):
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

    ###########################################################################
    def _instantiate_parameters(self, input_shape, input_dtype):
        _LG.debug('    Input: shape %s, dtype %s', input_shape, input_dtype)
        initializers = _get_initializers(
            self.args.get('initializers') or {}, self.args['with_bias'])

        w_shape = self._get_weight_shape(input_shape)
        self._check_filter_shape(input_shape, w_shape)
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


class ReLU(LayerMixin, base_layer.BaseReLU):
    """Implement ReLU in Tensorflow.

    See :any:`BaseReLU` for detail.
    """
    def _build(self, input_tensor):
        output = tf.nn.relu(input_tensor.unwrap(), 'ouptut')
        return wrapper.Tensor(output, name='output')


class Sigmoid(LayerMixin, base_layer.BaseSigmoid):
    """Implement Sigmoid in Tensorflow.

    See :any:`BaseSigmoid` for detail.
    """
    def _build(self, input_tensor):
        output = tf.sigmoid(input_tensor.unwrap(), 'output')
        return wrapper.Tensor(output, name='output')


class Tanh(LayerMixin, base_layer.BaseTanh):
    """Implement Tanh in Tensorflow.

    See :any:`BaseTanh` for detail.
    """
    def _build(self, input_tensor):
        output = tf.tanh(input_tensor.unwrap(), 'output')
        return wrapper.Tensor(output, name='output')


class Sin(LayerMixin, base_layer.BaseSin):
    """Implement Sin in Tensorflow.

    See :any:`BaseSin` for detail.
    """
    def _build(self, input_tensor):
        output = tf.sin(input_tensor.unwrap(), 'output')
        return wrapper.Tensor(output, name='output')


class Cos(LayerMixin, base_layer.BaseCos):
    """Implement Cos in Tensorflow.

    See :any:`BaseCos` for detail.
    """
    def _build(self, input_tensor):
        output = tf.cos(input_tensor.unwrap(), 'output')
        return wrapper.Tensor(output, name='output')


class Softmax(LayerMixin, base_layer.BaseSoftmax):
    """Implement Softmax in Tensorflow.

    See :any:`BaseSoftmax` for detail.
    """
    def _build(self, input_tensor):
        output = tf.nn.softmax(input_tensor.unwrap())
        return wrapper.Tensor(output, name='output')


class Softplus(LayerMixin, base_layer.BaseSoftplus):
    """Implement Softplus in Tensorflow.

    See :any:`BaseSoftplus` for detail.
    """
    def _build(self, input_tensor):
        output = tf.nn.softplus(input_tensor.unwrap())
        return wrapper.Tensor(output, name='output')


###############################################################################
class Flatten(LayerMixin, base_layer.BaseFlatten):
    """Implement Flatten in Tensorflow.

    See :any:`BaseFlatten` for detail.
    """
    def _build(self, input_tensor):
        in_shape = input_tensor.shape
        n_nodes = reduce(lambda prod, dim: prod*dim, in_shape[1:], 1)
        out_shape = (-1, n_nodes)
        output = tf.reshape(input_tensor.unwrap(), out_shape, 'output')
        return wrapper.Tensor(output, name='output')


class Tile(LayerMixin, base_layer.BaseTile):
    """Implement Tile layer in Tensorflow

    See :any:`BaseFlatten` for detail.
    """
    def _build(self, input_tensor):
        return input_tensor.tile(self.args['pattern'], name='output')


###############################################################################
class Concat(LayerMixin, base_layer.BaseConcat):
    """Implement Concat in Tensorflow.

    See :any:`BaseConcat` for detail.
    """
    def _build(self, var_list):
        values = [var.unwrap() for var in var_list]
        output = tf.concat(values, axis=self.args['axis'])
        return wrapper.Tensor(output, name='output')


class Add(LayerMixin, base_layer.BaseAdd):
    """Implement Add layer in Tensorflow

    See :any: `BaseAdd` for detail.
    """
    def _build(self, var_list):
        if len(var_list) < 2:
            raise ValueError('var_list must contain at least 2 tensors')

        ret = var_list[0]
        for var in var_list[1:-1]:
            ret = ret + var
        return ret.__add__(var_list[-1], name='output')


class Sub(LayerMixin, base_layer.BaseAdd):
    """Implement Sub layer in Tensorflow

    See :any: `BaseSub` for detail.
    """
    def _build(self, var_list):
        if len(var_list) != 2:
            raise ValueError('var_list must be 2 tensors')

        return var_list[0].__sub__(var_list[1], name='output')


###############################################################################
class TrueDiv(LayerMixin, base_layer.BaseTrueDiv):
    """Implement TrueDiv in Tensorflow.

    See :any:`BaseTrueDiv` for detail.
    """
    def _instantiate_denominator(self, dtype):
        self.denom = tf.constant(
            self.args['denom'], dtype=dtype, name='denominator')

    def _build(self, input_tensor):
        dtype = input_tensor.dtype
        tensor = input_tensor.unwrap()
        if 'int' in input_tensor.dtype:
            dtype = luchador.get_nn_dtype()
            tensor = tf.cast(tensor, dtype)

        if self.denom is None:
            self._instantiate_denominator(dtype)

        output = tf.truediv(tensor, self.denom, 'ouptut')
        return wrapper.Tensor(output, name='output')


class Mean(LayerMixin, base_layer.BaseMean):
    """Implement Mean layer in Tensorflow.

    See :any:`BaseMean` for detail.
    """
    def _build(self, input_tensor):
        return input_tensor.mean(name='output', **self.args)


###############################################################################
class BatchNormalization(LayerMixin, base_layer.BaseBatchNormalization):
    """Implement BatchNormalization in Tensorflow.

    See :any:`BaseBatchNormalization` for detail.
    """
    def _instantiate_parameters(self, input_shape):
        dim, fmt = len(input_shape), luchador.get_nn_conv_format()
        channel = 1 if dim == 2 or fmt == 'NCHW' else 3

        self._axes = tuple(i for i in range(dim) if not i == channel)
        shape = tuple(input_shape[i] for i in range(dim) if i == channel)

        mean = scope.get_variable(
            name='mean', shape=shape,
            initializer=initializer.Constant(0), trainable=False)
        var = scope.get_variable(
            name='var', shape=shape,
            initializer=initializer.Constant(1), trainable=False)

        scale = scope.get_variable(
            name='scale', shape=shape, trainable=True,
            initializer=initializer.Constant(self.args['scale']))
        offset = scope.get_variable(
            name='offset', shape=shape, trainable=True,
            initializer=initializer.Constant(self.args['offset']))

        self._add_parameter('mean', mean)
        self._add_parameter('var', var)
        self._add_parameter('scale', scale)
        self._add_parameter('offset', offset)

    def _build(self, input_tensor):
        input_shape = input_tensor.shape
        if not self._parameter_variables:
            self._instantiate_parameters(input_shape)

        input_ = input_tensor.unwrap()
        decay, epsilon = self.args['decay'], self.args['epsilon']

        mean_acc = self._get_parameter('mean').unwrap()
        var_acc = self._get_parameter('var').unwrap()
        scale = self._get_parameter('scale').unwrap()
        offset = self._get_parameter('offset').unwrap()

        if self.args['learn']:
            mean_in, var_in = tf.nn.moments(input_, self._axes)

            new_mean_acc = decay * mean_acc + (1 - decay) * mean_in
            new_var_acc = decay * var_acc + (1 - decay) * var_in

            self._add_update('mean', tf.assign(mean_acc, new_mean_acc))
            self._add_update('var', tf.assign(var_acc, new_var_acc))

            mean_acc = new_mean_acc
            var_acc = new_var_acc

        output = tf.nn.batch_normalization(
            x=input_, mean=mean_acc, variance=var_acc, offset=offset,
            scale=scale, variance_epsilon=epsilon)
        return wrapper.Tensor(output, name='output')


###############################################################################
class NHWC2NCHW(LayerMixin, base_layer.BaseNHWC2NCHW):
    """See :any:`BaseNHWC2NCHW` for detail."""
    def _build(self, input_tensor):
        output = tf.transpose(input_tensor.unwrap(), perm=(0, 3, 1, 2))
        return wrapper.Tensor(output, name='output')


class NCHW2NHWC(LayerMixin, base_layer.BaseNCHW2NHWC):
    """See :any:`BaseNCHW2NHWC` for detail."""
    def _build(self, input_tensor):
        output = tf.transpose(input_tensor.unwrap(), perm=(0, 2, 3, 1))
        return wrapper.Tensor(output, name='output')
