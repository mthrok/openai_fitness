"""Implement Layer classes in Theano"""
from __future__ import division
from __future__ import absolute_import

import logging
import warnings

import theano.tensor as T
from theano.tensor.nnet.abstract_conv import get_conv_output_shape

from ... import common
from ...base.getter import get_initializer
from ...base.layer import BaseConv2D, BaseConv2DTranspose
from .. import wrapper

__all__ = ['Conv2D', 'Conv2DTranspose']

_LG = logging.getLogger(__name__)


def _map_border_mode(padding):
    if isinstance(padding, str):
        mode = padding.lower()
        return 'half' if mode == 'same' else mode
    return padding


def _is_int_list(list_, length=2):
    return len(list_) == length and all([isinstance(e, int) for e in list_])


def _validate_padding(padding):
    msg = ('`padding` must be either str ("valid", "full", "half" or '
           '"same"), int or tuple of two int')

    if isinstance(padding, int):
        return

    if isinstance(padding, str):
        if padding.lower() in ['full', 'half', 'same', 'valid']:
            return
        raise ValueError(msg)

    try:
        if _is_int_list(padding, length=2):
            return
    except TypeError:
        pass

    raise ValueError(msg)


def _validate_strides(strides):
    if isinstance(strides, int):
        return
    try:
        if _is_int_list(strides, length=2):
            return
    except TypeError:
        pass

    raise ValueError('`strides` must be either int or tuple of two int')


def _check_output_shape(input_shape, filter_shape, border_mode, subsample):
    """Issue warning if a part of image is not covered by filter"""
    f_row, f_col = filter_shape[2:4]
    in_row, in_col = input_shape[2:4]
    sub_row, sub_col = subsample
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
        warn_row = f_row < sub_row
        warn_col = f_col < sub_col
    else:
        warn_row = bool((in_row - f_row) % sub_row)
        warn_col = bool((in_col - f_col) % sub_col)
    if warn_col:
        warnings.warn(
            'Convolution op will not cover the right side of the input.'
            'Check the width configuration of filter and stride.',
            RuntimeWarning
        )
    if warn_row:
        warnings.warn(
            'Convolution op will not cover the bottom part of the input.'
            'Check the height configuration of filter and stride.',
            RuntimeWarning
        )


def _get_subsample(strides):
    if isinstance(strides, int):
        return (strides, strides)
    return strides


def _get_filter_init(config):
    """Make filter initializer. Default to Xavier"""
    config = config or {'typename': 'Xavier'}
    return get_initializer(config['typename'])(**config.get('args', {}))


def _get_bias_init(config):
    """Make bias initializer. Default to Constant (0.1)"""
    config = config or {'typename': 'Constant', 'args': {'value': 0.1}}
    return get_initializer(config['typename'])(**config.get('args', {}))


class _Conv2DMixin(object):
    # pylint: disable=no-self-use, too-few-public-methods
    def _validate_args(self, padding, strides, **_):
        _validate_padding(padding)
        _validate_strides(strides)

    def _build_filter(self, shape, dtype):
        init = _get_filter_init(self.args['initializers'].get('filter'))
        filter_ = wrapper.get_variable(
            name='filter', shape=shape, initializer=init, dtype=dtype)
        self.set_parameter_variables(filter=filter_)

    def _build_bias(self, shape, dtype):
        init = _get_bias_init(self.args['initializers'].get('bias'))
        bias = wrapper.get_variable(
            name='bias', shape=shape, initializer=init, dtype=dtype)
        self.set_parameter_variables(bias=bias)

    def _build_parameters(self, filter_shape, bias_shape, dtype):
        if self._parameter_variables['filter'] is None:
            self._build_filter(shape=filter_shape, dtype=dtype)

        if not self.args['with_bias']:
            return

        if self._parameter_variables['bias'] is None:
            self._build_bias(shape=bias_shape, dtype=dtype)

    def _get_filter_shape(self, n_outputs):
        return (
            self.args['n_filters'], n_outputs,
            self.args['filter_height'], self.args['filter_width']
        )


class Conv2D(_Conv2DMixin, BaseConv2D):
    """Implement Conv2D layer in Theano.

    See :any:`BaseConv2D` for detail.
    """
    def _build(self, input_tensor):
        """Build 2D conolution operation of the input tensor

        Parameters
        ----------
        input_tensor : Tensor
            4D Tensor with shape (batch, #input channel, row, col)

        Returns
        -------
        Tensor
            4D Tensor with shape (batch, #output channel, row, col)
        """
        input_shape = input_tensor.shape
        _LG.debug('    input_shape: %s', input_shape)

        if not len(input_shape) == 4:
            raise ValueError(
                'Input tensor must be 4D. ({})'.format(input_tensor))

        border_mode = _map_border_mode(self.args['padding'])
        subsample = _get_subsample(self.args['strides'])
        filter_shape = self._get_filter_shape(input_shape[1])
        bias_shape = (filter_shape[0],)
        output_shape = get_conv_output_shape(
            input_shape, filter_shape, border_mode, subsample)
        _check_output_shape(input_shape, filter_shape, border_mode, subsample)

        _LG.debug('    border_mode: %s', border_mode)
        _LG.debug('    subsample: %s', subsample)
        _LG.debug('    filter_shape: %s', filter_shape)
        _LG.debug('    output_shape: %s', output_shape)

        self._build_parameters(filter_shape, bias_shape, input_tensor.dtype)

        filters = self.get_parameter_variables('filter')
        output_tensor = T.nnet.conv2d(
            input_tensor.unwrap(), filters=filters.unwrap(),
            input_shape=input_shape, filter_shape=filter_shape,
            border_mode=border_mode, subsample=subsample)

        if self.args['with_bias']:
            bias = self.get_parameter_variables('bias').unwrap()
            bias = bias.dimshuffle(('x', 0, 'x', 'x'))
            output_tensor = bias + output_tensor

        return wrapper.Tensor(output_tensor, shape=output_shape, name='output')


class Conv2DTranspose(_Conv2DMixin, BaseConv2DTranspose):
    """Implement Conv2DTranspose layer in Theano.

    See :any:`BaseConv2DTranspose` for detail.
    """
    def _get_output_shape_from_arg(self):
        if self.args.get('output_shape_format') == 'NHWC':
            _LG.info('  * Converting `output_shape` to NCHW')
            return common.nhwc2nchw(self.args['output_shape'])
        return self.args['output_shape']

    def _get_output_shape(self):
        if self.args['output_shape']:
            return self._get_output_shape_from_arg()
        if self._parameter_variables['original_input'] is not None:
            return self._parameter_variables['original_input'].shape
        raise RuntimeError(
            'Output shape is not given. Output shape must be given '
            'either as constructor `ouptut_shape` parameter or as '
            'parameter variable named `original_input` given via '
            '`set_parameter_variables` method.'
        )

    def _get_filter_shape(self, n_filters):
        if self.get_parameter_variables('filter') is not None:
            return self.get_parameter_variables('filter').shape
        if self.get_parameter_variables('original_filter') is not None:
            return self.get_parameter_variables('original_filter').shape
        return super(Conv2DTranspose, self)._get_filter_shape(n_filters)

    def _build(self, input_tensor):
        # In Theano, the notation of input and output is flipped because
        # they are re-using the terminology from gradient computation.
        output_shape = self._get_output_shape()

        filter_shape = self._get_filter_shape(output_shape[1])
        bias_shape = (filter_shape[1],)
        self._build_parameters(filter_shape, bias_shape, input_tensor.dtype)

        filters = self.get_parameter_variables('filter')
        border_mode = _map_border_mode(self.args['padding'])
        subsample = _get_subsample(self.args['strides'])
        tensor_ = T.nnet.abstract_conv.conv2d_grad_wrt_inputs(
            output_grad=input_tensor.unwrap(), filters=filters.unwrap(),
            input_shape=output_shape, filter_shape=filters.shape,
            border_mode=border_mode, subsample=subsample,
        )

        if self.args['with_bias']:
            bias = self.get_parameter_variables('bias').unwrap()
            tensor_ = bias.dimshuffle(('x', 0, 'x', 'x')) + tensor_
        return wrapper.Tensor(tensor_, shape=output_shape, name='output')
