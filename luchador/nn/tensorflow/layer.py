"""Implement Layer classes in Tensorflow"""
from __future__ import division
from __future__ import absolute_import

import logging
import warnings

import tensorflow as tf

import luchador
from luchador.nn.base import (
    layer as base_layer,
    initializer as base_initializer,
)
from . import scope, wrapper, initializer

# pylint: disable=too-few-public-methods, invalid-name

__all__ = [
    'LayerMixin',
    'Dense', 'Conv2D',
    'ReLU', 'Sigmoid', 'Softmax',
    'Flatten', 'Concat', 'TrueDiv',
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


def _wrap_output(tensor, name='output'):
    """Prefix the name of output tensor with current scope"""
    name = '{}/{}'.format(tf.get_variable_scope().name, name)
    return wrapper.Tensor(tensor, name=name)


def _get_initializers(cfg, with_bias):
    """Initializer for Dense and Conv2D"""
    w_cfg = cfg.get('weight')
    ret = {}
    ret['weight'] = (
        base_initializer.get_initializer(w_cfg['name'])(**w_cfg['args'])
        if w_cfg else initializer.Xavier()
    )

    if with_bias:
        b_cfg = cfg.get('bias')
        ret['bias'] = (
            base_initializer.get_initializer(b_cfg['name'])(**b_cfg['args'])
            if b_cfg else initializer.Constant(0.1)
        )

    return ret


class Dense(LayerMixin, base_layer.BaseDense):
    """Implement Dense layer in Tensorflow.

    See :any:`BaseDense` for detail.
    """
    def _instantiate_parameters(self, n_inputs):
        initializers = _get_initializers(
            self.args.get('initializers') or {}, self.args['with_bias'])

        w_shape = (n_inputs, self.args['n_nodes'])
        w_init = initializers['weight']
        self._add_parameter('weight', scope.get_variable(
            name='weight', shape=w_shape, initializer=w_init))

        if self.args['with_bias']:
            b_shape = (self.args['n_nodes'],)
            b_init = initializers['bias']
            self._add_parameter('bias', scope.get_variable(
                name='bias', shape=b_shape, initializer=b_init))

    def _build(self, input_tensor):
        if not self.parameter_variables:
            self._instantiate_parameters(input_tensor.shape[1])

        weight = self._get_parameter('weight').unwrap()
        output = tf.matmul(input_tensor.unwrap(), weight)

        if self.args['with_bias']:
            bias = self._get_parameter('bias').unwrap()
            output = tf.add(output, bias, name='output')
        return _wrap_output(output)


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
    def _instantiate_parameters(self, input_shape):
        _LG.debug('    Input shape: %s', input_shape)
        initializers = _get_initializers(
            self.args.get('initializers') or {}, self.args['with_bias'])

        w_shape = self._get_weight_shape(input_shape)
        self._check_filter_shape(input_shape, w_shape)
        w_init = initializers['weight']
        self._add_parameter('weight', scope.get_variable(
            name='weight', shape=w_shape, initializer=w_init))

        if self.args['with_bias']:
            b_shape = (self.args['n_filters'],)
            b_init = initializers['bias']
            self._add_parameter('bias', scope.get_variable(
                name='bias', shape=b_shape, initializer=b_init))

    def _build(self, input_tensor):
        if not self.parameter_variables:
            self._instantiate_parameters(input_tensor.shape)

        weight = self._get_parameter('weight').unwrap()
        strides = self._get_strides()
        name = self.args.get('name')
        cudnn = self.args.get('use_cudnn_on_gpu', True)
        fmt = self._get_format()
        padding = self._get_padding()
        output_tensor = tf.nn.conv2d(
            input_tensor.unwrap(), weight, strides=strides,
            padding=padding, use_cudnn_on_gpu=cudnn,
            data_format=fmt, name=name)

        if self.args['with_bias']:
            bias = self._get_parameter('bias').unwrap()
            output_tensor = tf.nn.bias_add(
                output_tensor, bias, data_format=fmt, name='output')
        return _wrap_output(output_tensor)


class ReLU(LayerMixin, base_layer.BaseReLU):
    """Implement ReLU in Tensorflow.

    See :any:`BaseReLU` for detail.
    """
    def _build(self, input_tensor):
        output = tf.nn.relu(input_tensor.unwrap(), 'ouptut')
        return _wrap_output(output)


class Sigmoid(LayerMixin, base_layer.BaseSigmoid):
    """Implement Sigmoid in Tensorflow.

    See :any:`BaseSigmoid` for detail.
    """
    def _build(self, input_tensor):
        output = tf.sigmoid(input_tensor.unwrap(), 'output')
        return _wrap_output(output)


class Softmax(LayerMixin, base_layer.BaseSoftmax):
    """Implement Softmax in Tensorflow.

    See :any:`BaseSoftmax` for detail.
    """
    def _build(self, input_tensor):
        output = tf.nn.softmax(input_tensor.unwrap())
        return _wrap_output(output)


class Flatten(LayerMixin, base_layer.BaseFlatten):
    """Implement Flatten in Tensorflow.

    See :any:`BaseFlatten` for detail.
    """
    def _build(self, input_tensor):
        in_shape = input_tensor.shape
        n_nodes = reduce(lambda prod, dim: prod*dim, in_shape[1:], 1)
        out_shape = (-1, n_nodes)
        output = tf.reshape(input_tensor.unwrap(), out_shape, 'output')
        return _wrap_output(output)


class Concat(LayerMixin, base_layer.BaseConcat):
    """Implement Concat in Tensorflow.

    See :any:`BaseConcate` for detail.
    """
    def _build(self, var_list):
        values = [var.unwrap() for var in var_list]
        output = tf.concat_v2(values, axis=self.args['axis'])
        return _wrap_output(output)


class TrueDiv(LayerMixin, base_layer.BaseTrueDiv):
    """Implement TrueDiv in Tensorflow.

    See :any:`BaseTrueDiv` for detail.
    """
    def _instantiate_denominator(self):
        dtype = self.args['dtype'] or luchador.get_nn_dtype()
        self.denom = tf.constant(
            self.args['denom'], dtype=dtype, name='denominator')

    def _build(self, input_tensor):
        if self.denom is None:
            self._instantiate_denominator()
        output = tf.truediv(input_tensor.unwrap(), self.denom, 'ouptut')
        return _wrap_output(output)


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
        if not self.parameter_variables:
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
        return _wrap_output(output)


###############################################################################
class NHWC2NCHW(LayerMixin, base_layer.BaseNHWC2NCHW):
    """See :any:`BaseNHWC2NCHW` for detail."""
    def _build(self, input_tensor):
        input_tensor_ = input_tensor.unwrap()
        output_tensor_ = tf.transpose(input_tensor_, perm=(0, 3, 1, 2))
        return _wrap_output(output_tensor_)


class NCHW2NHWC(LayerMixin, base_layer.BaseNCHW2NHWC):
    """See :any:`BaseNCHW2NHWC` for detail."""
    def _build(self, input_tensor):
        input_tensor_ = input_tensor.unwrap()
        output_tensor_ = tf.transpose(input_tensor_, perm=(0, 2, 3, 1))
        return _wrap_output(output_tensor_)
