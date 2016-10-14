from __future__ import division
from __future__ import absolute_import

import logging

import theano
import theano.tensor as T

from ..base import (
    get_layer,
    get_initializer,
    Layer as BaseLayer,
)
from . import scope as scp
from .wrapper import (
    Tensor,
    Operation,
)
from .initializer import (
    Constant,
    Xavier,
    XavierConv2D,
)


_LG = logging.getLogger(__name__)

__all__ = [
    'BaseLayer', 'get_layer',
    'Dense', 'Conv2D', 'ReLU', 'Flatten', 'TrueDiv', 'BatchNormalization'
]


def _wrap_output(tensor, shape, name='output'):
    """Add scope prefix to the name of ouptut Tenaor"""
    name = '{}/{}'.format(scp.get_variable_scope().name, name)
    return Tensor(tensor, shape=shape, name=name)


class TheanoLayer(BaseLayer):
    def get_update_operation(self):
        return Operation(self.update_operations)


class Dense(TheanoLayer):
    def __init__(self, n_nodes, initializers={}, with_bias=True):
        """Initialize dense layer.
        Activation function, such as ReLU is not included.
        Also called fully connected, affine, linear or inner product.

        Args:
          n_nodes (int): The number of internal neurons.
          initializers (dict): Dictionary containing configuration.
          with_bias (bool): When True bias term is added to graph
        """
        super(Dense, self).__init__(
            n_nodes=n_nodes, initializers=initializers, with_bias=with_bias)

    def _instantiate_initializers(self):
        init_cfg = self.args.get('initializers', {})

        cfg = init_cfg.get('weight')
        self.initializers['weight'] = (
            get_initializer(cfg['name'])(**cfg['args'])
            if cfg else Xavier()
        )

        if self.args['with_bias']:
            cfg = init_cfg.get('bias')
            self.initializers['bias'] = (
                get_initializer(cfg['name'])(**cfg['args'])
                if cfg else Constant(0.1)
            )

    def _instantiate_parameter_variables(self, n_inputs):
        self._instantiate_initializers()

        w_shape = (n_inputs, self.args['n_nodes'])
        w_init = self.initializers['weight']
        self._add_parameter('weight', scp.get_variable(
            name='weight', shape=w_shape, initializer=w_init))

        if self.args['with_bias']:
            b_shape = (self.args['n_nodes'],)
            b_init = self.initializers['bias']
            self._add_parameter('bias', scp.get_variable(
                name='bias', shape=b_shape, initializer=b_init))

    def build(self, input_tensor):
        """
        Args:
          input_tensor (TensorWrapper): 2D tensor

        Returns:
          TensorWrapper: 2D tensor wrapper
        """
        _LG.debug('    Building {}: {}'.format(type(self).__name__, self.args))
        input_shape = input_tensor.get_shape()

        if not len(input_shape) == 2:
            raise ValueError('Input tensor must be 2D. '
                             'Insted of {}'.format(len(input_shape)))

        if not self.parameter_variables:
            self._instantiate_parameter_variables(input_shape[1])

        weight = self._get_parameter('weight').unwrap()
        output_tensor = T.dot(input_tensor.unwrap(), weight)

        if self.args['with_bias']:
            bias = self._get_parameter('bias').unwrap()
            output_tensor = output_tensor + bias
        output_shape = (input_shape[0], self.args['n_nodes'])
        return _wrap_output(output_tensor, output_shape, 'output')


def _map_border_mode(padding):
    if isinstance(padding, str):
        mode = padding.lower()
        return 'half' if mode == 'same' else mode
    return padding


class Conv2D(TheanoLayer):
    """Apply convolution to input"""
    def __init__(self, filter_height, filter_width, n_filters, strides,
                 padding='VALID', initializers={}, with_bias=True, **kwargs):
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

          initializers (dict): Dictionary containing configuration.

          with_bias (bool): When True bias term is added to graph

          kwargs:
            - Tensorflow: Arguments passed to tf.nn.conv2d.
              'use_cudnn_on_gpu' and 'name'
        """
        super(Conv2D, self).__init__(
            filter_height=filter_height, filter_width=filter_width,
            n_filters=n_filters, strides=strides, padding=padding,
            initializers=initializers, with_bias=with_bias, **kwargs)

    ###########################################################################
    # Parameter validation
    def _validate_padding(self, padding):
        msg = ('`padding` must be either str ("valid", "full", "half" or '
               '"same"), int or tuple of two int')

        if isinstance(padding, int):
            return

        if isinstance(padding, str):
            if padding.lower() in ['full', 'half', 'same', 'valid']:
                return
            raise ValueError(msg)

        try:
            p0, p1 = padding[0], padding[1]
            if (isinstance(p0, int) and isinstance(p1, int)):
                return
        except Exception:
            pass

        raise ValueError(msg)

    def _validate_strides(self, strides):
        if isinstance(strides, int):
            return
        try:
            p0, p1 = strides[0], strides[1]
            if (isinstance(p0, int) and isinstance(p1, int)):
                return
        except Exception:
            pass

        raise ValueError('`strides` must be either int or tuple of two int')

    def _validate_args(self, args):
        self._validate_padding(args['padding'])
        self._validate_strides(args['strides'])

    ###########################################################################
    def _instantiate_initializers(self):
        init_cfg = self.args.get('initializers', {})

        cfg = init_cfg.get('weight')
        self.initializers['weight'] = (
            get_initializer(cfg['name'])(**cfg['args'])
            if cfg else XavierConv2D()
        )

        if self.args['with_bias']:
            cfg = init_cfg.get('bias')
            self.initializers['bias'] = (
                get_initializer(cfg['name'])(**cfg['args'])
                if cfg else Constant(0.1)
            )

    def _instantiate_parameter_variables(self, n_inputs):
        self._instantiate_initializers()

        w_shape = (self.args['n_filters'], n_inputs,
                   self.args['filter_height'], self.args['filter_width'])
        w_init = self.initializers['weight']
        self._add_parameter('weight', scp.get_variable(
            name='weight', shape=w_shape, initializer=w_init))

        if self.args['with_bias']:
            b_shape = (self.args['n_filters'],)
            b_init = self.initializers['bias']
            self._add_parameter('bias', scp.get_variable(
                name='bias', shape=b_shape, initializer=b_init))

    def _get_subsample(self):
        if isinstance(self.args['strides'], int):
            return (self.args['strides'], self.args['strides'])
        return self.args['strides']

    def _get_border_mode(self):
        return _map_border_mode(self.args['padding'])

    def _get_output_shape(self, input_shape, filter_shape):
        """Compute output shape

        Args:
          input_shape(tuple): (batch, n_input_channels, row, col)
          filter_shape(tuple): (n_filters, n_input_channels, rows, cols)
        """
        # TODO: Add warning if
        # parts of image are not covered because of subsampling
        f_row, f_col = filter_shape[2:4]
        in_row, in_col = input_shape[2:4]
        sub_row, sub_col = self._get_subsample()
        border_mode = self._get_border_mode()
        # Process padding
        if border_mode in ['full', 'valid']:
            pass
        elif border_mode == 'half':
            in_row += 2 * (f_row // 2)
            in_col += 2 * (f_col // 2)
        elif isinstance(border_mode, int):
            in_row += 2 * border_mode
            in_col += 2 * border_mode
        else:
            in_row += 2 * border_mode[0]
            in_col += 2 * border_mode[1]
        # Process convolution
        if border_mode == 'full':
            out_row = (in_row + f_row - 2) // sub_row + 1
            out_col = (in_col + f_col - 2) // sub_col + 1
        else:
            out_row = (in_row - f_row) // sub_row + 1
            out_col = (in_col - f_col) // sub_col + 1
        # Reconstruct
        n_batches, n_filters = input_shape[0], filter_shape[0]
        output_shape = (n_batches, n_filters, out_row, out_col)
        return output_shape

    def build(self, input_tensor):
        """Build 2D conolution on top of the input tensor
        Args:
          input_tensor (TensorWrapper):
            4D Tensor with shape (batch, channel, row, col).

        Returns:
          TensorWrapper: 4D Tensor with shape (batch, stack, row, col)
        """
        input_shape = input_tensor.get_shape()

        _LG.debug('    Building {}: {}'.format(type(self).__name__, self.args))
        _LG.debug('    input_shape: {}'.format(input_shape))
        _LG.debug('    border_mode: {}'.format(self._get_border_mode()))

        if not len(input_shape) == 4:
            raise ValueError('Input tensor must be 4D. '
                             'Insted of {}'.format(len(input_shape)))

        if not self.parameter_variables:
            self._instantiate_parameter_variables(input_shape[1])

        filters = self._get_parameter('weight').unwrap()
        filter_shape = filters.get_value().shape
        subsample = self._get_subsample()
        border_mode = self._get_border_mode()

        output_tensor = T.nnet.conv2d(
            input_tensor.unwrap(), filters=filters,
            input_shape=input_shape, filter_shape=filter_shape,
            border_mode=border_mode, subsample=subsample)

        if self.args['with_bias']:
            bias = self._get_parameter('bias').unwrap()
            bias = bias.dimshuffle(('x', 0, 'x', 'x'))
            output_tensor = bias + output_tensor

        output_shape = self._get_output_shape(input_shape, filter_shape)
        _LG.debug('    output_shape: {}'.format(output_shape))
        return _wrap_output(output_tensor, output_shape, 'output')


class ReLU(TheanoLayer):
    """Applies Rectified Linear Unit"""
    def __init__(self):
        super(ReLU, self).__init__()

    def build(self, input_tensor):
        """
        Args:
          input_tensor (ShapedTensor): Placeholder object.
          Output: Output object
        """
        input_shape = input_tensor.get_shape()
        _LG.debug('    Building {}: {}'.format(type(self).__name__, self.args))
        _LG.debug('    input_shape: {}'.format(input_shape))

        output_tensor = T.nnet.relu(input_tensor.unwrap())
        return _wrap_output(output_tensor, input_shape, name='output')


class Flatten(TheanoLayer):
    """Reshape batch into 2D (batch_size, n_features)"""
    def __init__(self):
        super(Flatten, self).__init__()

    def build(self, input_tensor):
        input_shape = input_tensor.get_shape()
        n_nodes = int(reduce(lambda r, d: r*d, input_shape[1:], 1))

        _LG.debug('    Building {}: {}'.format(type(self).__name__, self.args))
        _LG.debug('    Input shape: {}'.format(input_shape))
        _LG.debug('    #Nodes     : {}'.format(n_nodes))

        output_shape = (input_shape[0] or -1, n_nodes)
        output_tensor = T.reshape(input_tensor.unwrap(), output_shape)
        _LG.debug('    output_shape: {}'.format(output_shape))
        return _wrap_output(output_tensor, output_shape, 'output')


class TrueDiv(TheanoLayer):
    """Applies element wise division"""
    def __init__(self, denom, dtype=None):
        super(TrueDiv, self).__init__(denom=denom, dtype=None)
        self.denom = None

    def _instantiate_denominator(self):
        dtype = self.args['dtype'] or theano.config.floatX
        self.denom = T.constant(
            self.args['denom'], dtype=dtype, name='denominator')

    def build(self, input_tensor):
        _LG.debug('    Building {}: {}'.format(type(self).__name__, self.args))
        if self.denom is None:
            self._instantiate_denominator()
        output_tensor = input_tensor.unwrap() / self.args['denom']
        return _wrap_output(output_tensor, input_tensor.get_shape(), 'output')


class BatchNormalization(TheanoLayer):
    """Applies batch normalization

    Ioffe, Sergey and Szegedy, Christian (2015):
           Batch Normalization: Accelerating Deep Network Training by Reducing
           Internal Covariate Shift. http://arxiv.org/abs/1502.03167.
    """
    def __init__(self, scale=1.0, center=0.0, epsilon=1e-4,
                 learn=True, decay=0.999):
        super(BatchNormalization, self).__init__(
            decay=decay, epsilon=epsilon,
            scale=scale, center=center, learn=learn)

    def _instantiate_parameter_variables(self, input_shape):
        """Instantiate variable for mean and standard diviation"""
        dim = len(input_shape)
        self.axes = tuple(i for i in range(dim) if not i == 1)
        self.shape = tuple(input_shape[i] for i in range(dim) if i == 1)
        self.pattern = tuple((0 if i == 1 else 'x') for i in range(dim))

        _LG.debug('    Shape: {}'.format(self.shape))
        _LG.debug('     Axes: {}'.format(self.axes))
        _LG.debug('  Pattern: {}'.format(self.pattern))

        mean = scp.get_variable(name='mean', shape=self.shape,
                                initializer=Constant(0), trainable=False)
        inv_std = scp.get_variable(name='inv_std', shape=self.shape,
                                   initializer=Constant(1), trainable=False)

        scale_, center_ = self.args['scale'], self.args['center']
        scale = scp.get_variable(
            name='scale', shape=self.shape,
            initializer=Constant(scale_), trainable=True)
        center = scp.get_variable(
            name='center', shape=self.shape,
            initializer=Constant(center_), trainable=True)

        self._add_parameter('mean', mean)
        self._add_parameter('inv_std', inv_std)
        self._add_parameter('scale', scale)
        self._add_parameter('center', center)

    def build(self, input_tensor):
        _LG.debug('    Building {}: {}'.format(type(self).__name__, self.args))
        if not self.parameter_variables:
            self._instantiate_parameter_variables(input_tensor.get_shape())

        input_tensor_ = input_tensor.unwrap()
        decay, ep = self.args['decay'], self.args['epsilon']

        mean_acc = self._get_parameter('mean').unwrap()
        stdi_acc = self._get_parameter('inv_std').unwrap()
        scale = self._get_parameter('scale').unwrap()
        center = self._get_parameter('center').unwrap()

        if self.args['learn']:
            mean_in = input_tensor_.mean(axis=self.axes)
            stdi_in = T.inv(T.sqrt(input_tensor_.var(self.axes) + ep))

            new_mean_acc = decay * mean_acc + (1 - decay) * mean_in
            new_stdi_acc = decay * stdi_acc + (1 - decay) * stdi_in

            self._add_update(mean_acc, new_mean_acc)
            self._add_update(stdi_acc, new_stdi_acc)

            mean_acc = new_mean_acc
            stdi_acc = new_stdi_acc

        mean_acc = mean_acc.dimshuffle(self.pattern)
        stdi_acc = stdi_acc.dimshuffle(self.pattern)
        scale = scale.dimshuffle(self.pattern)
        center = center.dimshuffle(self.pattern)

        output = scale * (input_tensor_ - mean_acc) * stdi_acc + center
        return _wrap_output(output, input_tensor.shape, 'output')
