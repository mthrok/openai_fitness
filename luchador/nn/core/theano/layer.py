from __future__ import division
from __future__ import absolute_import

import logging

import theano
import theano.tensor as T

from ..base import (
    get_layer,
    get_initializer,
    Layer as BaseLayer,
    ReLU as BaseReLU,
    Dense as BaseDense,
    Conv2D as BaseConv2D,
    Flatten as BaseFlatten,
    TrueDiv as BaseTrueDiv,
)
from . import scope as scp
from .wrapper import Tensor
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


def _wrap_output(tensor, shape, name='output'):
    name = '{}/{}'.format(scp.get_variable_scope().name, name)
    return Tensor(tensor, shape=shape, name=name)


class Dense(BaseDense):
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

        args = self.args
        b_shape = (args['n_nodes'], )
        w_shape = (n_inputs, args['n_nodes'])

        b_init = self.initializers['bias']
        w_init = self.initializers['weight']

        self.parameter_variables['weight'] = scp.get_variable(
            name='weight', shape=w_shape, initializer=w_init)
        self.parameter_variables['bias'] = scp.get_variable(
            name='bias', shape=b_shape, initializer=b_init)

    def build(self, input):
        """
        Args:
          input (TensorWrapper): 2D tensor

        Returns:
          TensorWrapper: 2D tensor wrapper
        """
        _LG.debug('    Building {}: {}'.format(type(self).__name__, self.args))
        if not len(input.shape) == 2:
            raise ValueError('Input tensor must be 2D. '
                             'Insted of {}'.format(len(input.shape)))

        if not self.parameter_variables:
            self._instantiate_parameter_variables(input.shape[1])

        prod = T.dot(input.get(), self.parameter_variables['weight'].get())
        output_tensor = prod + self.parameter_variables['bias'].get()
        output_shape = (input.shape[0], self.args['n_nodes'])
        return _wrap_output(output_tensor, output_shape, 'output')


def _map_border_mode(padding):
    if isinstance(padding, str):
        mode = padding.lower()
        return 'half' if mode == 'same' else mode
    return padding


class Conv2D(BaseConv2D):
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

    def _instantiate_parameter_variables(self, n_inputs):
        self._instantiate_initializers()

        args = self.args
        b_shape = (args['n_filters'],)
        w_shape = (args['n_filters'], n_inputs,
                   args['filter_height'], args['filter_width'])

        b_init = self.initializers['bias']
        w_init = self.initializers['weight']

        self.parameter_variables['weight'] = scp.get_variable(
            name='weight', shape=w_shape, initializer=w_init)
        self.parameter_variables['bias'] = scp.get_variable(
            name='bias', shape=b_shape, initializer=b_init)

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

    def build(self, input):
        """Build 2D conolution on top of the input tensor
        Args:
          input (TensorWrapper): 4D Tensor with shape (batch, stack, row, col).

        Returns:
          TensorWrapper: 4D Tensor with shape (batch, stack, row, col)
        """
        _LG.debug('    Building {}: {}'.format(type(self).__name__, self.args))
        _LG.debug('    input_shape: {}'.format(input.shape))
        _LG.debug('    border_mode: {}'.format(self._get_border_mode()))
        if not len(input.shape) == 4:
            raise ValueError('Input tensor must be 4D. '
                             'Insted of {}'.format(len(input.shape)))

        if not self.parameter_variables:
            self._instantiate_parameter_variables(input.shape[1])

        filters = self.parameter_variables['weight'].get()
        filter_shape = filters.get_value().shape
        subsample = self._get_subsample()
        border_mode = self._get_border_mode()

        conv = T.nnet.conv2d(
            input.get(), filters=filters,
            input_shape=input.shape, filter_shape=filter_shape,
            border_mode=border_mode, subsample=subsample)

        bias = self.parameter_variables['bias'].get()
        bias = bias.dimshuffle(('x', 0, 'x', 'x'))
        output_tensor = bias + conv
        output_shape = self._get_output_shape(input.shape, filter_shape)
        _LG.debug('    output_shape: {}'.format(output_shape))
        return _wrap_output(output_tensor, output_shape, 'output')


class ReLU(BaseReLU):
    def build(self, input):
        """
        Args:
          input (ShapedTensor): Placeholder object.
          Output: Output object
        """
        _LG.debug('    Building {}: {}'.format(type(self).__name__, self.args))
        output_tensor = T.nnet.relu(input.get())
        _LG.debug('    input_shape: {}'.format(input.shape))
        return _wrap_output(output_tensor, input.shape, name='output')


class Flatten(BaseFlatten):
    def build(self, input):
        _LG.debug('    Building {}: {}'.format(type(self).__name__, self.args))
        _LG.debug('    Input shape: {}'.format(input.shape))
        n_nodes = int(reduce(lambda r, d: r*d, input.shape[1:], 1))
        _LG.debug('    #Nodes     : {}'.format(n_nodes))
        output_shape = (input.shape[0] or -1, n_nodes)
        output_tensor = T.reshape(input.get(), output_shape)
        _LG.debug('    output_shape: {}'.format(output_shape))
        return _wrap_output(output_tensor, output_shape, 'output')


class TrueDiv(BaseTrueDiv):
    def _instantiate_denominator(self):
        dtype = self.args['dtype'] or theano.config.floatX
        self.denom = T.constant(
            self.args['denom'], dtype=dtype, name='denominator')

    def build(self, input):
        _LG.debug('    Building {}: {}'.format(type(self).__name__, self.args))
        if self.denom is None:
            self._instantiate_denominator()
        output_tensor = input.get() / self.args['denom']
        return _wrap_output(output_tensor, input.shape, 'output')
