"""Define common interface for Layer classes"""
from __future__ import division
from __future__ import absolute_import

import logging

from .base import BaseLayer

__all__ = ['BaseConv2D']

_LG = logging.getLogger(__name__)

# pylint: disable=abstract-method


class BaseConv2D(BaseLayer):
    """Apply 2D convolution.

    Input Tensor : 4D tensor
        NCHW Format
            (batch size, **#input channels**, input height, input width)

        NHWC format : (Tensorflow backend only)
            (batch size, input height, input width, **#input channels**)

    Output Shape
        NCHW Format
            (batch size, **#output channels**, output height, output width)

        NHWC format : (Tensorflow backend only)
            (batch size, output height, output width, **#output channels**)

    Parameters
    ----------
    filter_height : int
        filter height, (#rows in filter)

    filter_width : int
        filter width (#columns in filter)

    n_filters : int
        #filters (#output channels)

    strides : (int, tuple of two ints, or tuple of four ints)
        ** When given type is int **
            The output is subsampled by this factor in both width and
            height direction.

        ** When given type is tuple of two int **
            The output is subsapmled by ``strides[0]`` in height and
            ``striders[1]`` in width.

        Notes
            [Tensorflow only]

            When given type is tuple of four int, their order must be
            consistent with the input data format.

            **NHWC**: (batch, height, width, channel)

            **NCHW**: (batch, channel, height, width)

    padding : (str or int or tuple of two ints)
        - Tensorflow : Either 'SAME' or 'VALID'
        - Theano : See doc for `theano.tensor.nnet.conv2d`

    with_bias : bool
        When True bias term is added after convolution

    kwargs
        use_cudnn_on_gpu
            [Tensorflow only] : Arguments passed to ``tf.nn.conv2d``

    Notes
    -----
    To fetch paramter variables with :any:`get_variable`, use keys
    ``filter`` and ``bias`` in the same scope as layer build.
    """
    def __init__(
            self, filter_height, filter_width, n_filters, strides,
            padding='VALID', initializers=None, with_bias=True,
            **kwargs):
        super(BaseConv2D, self).__init__(
            filter_height=filter_height, filter_width=filter_width,
            n_filters=n_filters, strides=strides, padding=padding,
            initializers=initializers or {}, with_bias=with_bias,
            **kwargs)

        keys = ['filter', 'bias'] if with_bias else ['filter']
        self._create_parameter_slots(*keys)
