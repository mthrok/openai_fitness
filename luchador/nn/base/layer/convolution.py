"""Define common interface for Layer classes"""
from __future__ import division
from __future__ import absolute_import

import logging

from .base import BaseLayer

__all__ = ['BaseConv2D', 'BaseConv2DTranspose']

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

    initializers: dict
        bias : dict
            Bias initializer configurations
        filter : dict
            Filter initializer configurations
    kwargs
        use_cudnn_on_gpu
            [Tensorflow only] : Arguments passed to ``tf.nn.conv2d``

    with_bias : bool
        When True bias term is added after convolution

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

        self._create_parameter_slot('filter', train=True, serialize=True)
        if with_bias:
            self._create_parameter_slot('bias', train=True, serialize=True)


class BaseConv2DTranspose(BaseLayer):
    """Upsample 2D array with reversed convolution (gradient of convolution)

    # TODO Add example pattern to create variable

    Internally (both Tensorflow, Theano), this is a convolution. Thus
    construction of this layer is a bit different from layer, thus you need
    to reuse filter variable from Conv2D.

    Examples
    --------
    >>> # Create convolution layer
    >>> conv2d = nn.layer.Conv2D(
    >>>     filter_height=7, filter_width=5, n_filters=3,
    >>>     strides=3, padding='valid')
    >>> input = nn.Input(shape=(32, 4, 84, 84))
    >>> conv_output = conv2d(input)

    >>> # Create convolution transpose layer with the same `padding` and
    >>> # `strides` parameters as the original convoultion layer,
    >>> conv2d_t = nn.layer.Conv2DTranspose(
    >>>     strides=3, padding='valid')

    >>> # Set `filter` and `original_input`. The output shape of transposed
    >>> # Convolution cannot be uniquely determined.
    >>> filter_var = conv2d.get_parameter_variables('filter')
    >>> conv2d_t.set_parameter_variables(
    >>>     filter=filter_var, original_input=input)
    >>> output = conv2d_t(conv_output)

    If you know the output size, you can give it as constructor argument and
    need not to set ``original_input``. You still need to set ``filter``.

    :py:func:`luchador.nn.util.model_maker.make_model` function can handle this
    by adding ``parameters``.

    .. code-block:: YAML

        typename: Conv2DTranspose
        args:
            strides: 3
            padding: VALID
            with_bias: True
        parameters:
            filter:
                typename: Variable
                name: layer1/filter
            original_input:
                typename: Input
                reuse: True
                name: input

    Parameters
    ----------
    _sentinel: Used to force the usage of keyward argument.

    filter_height, filter_width : int
        The shape of filter. Only required when not reusing an existing
        filter Variable. This should be the same value as corresponding Conv2D
        layer.

    n_filters : int
        The input channel of filter. Only required when not reusing an existing
        filter Variable. This should be the same value as corresponding Conv2D
        layer.

    strides : (int, tuple of two ints, or tuple of four ints)
        Not optional. See :any:`BaseConv2D`. This has to consistent with input
        shape and output shape.

    padding : (str or int or tuple of two ints)
        Not optional. See :any:`BaseConv2D`. This has to consistent with input
        shape and output shape.

    initializers: dict
        Initializer configurations. See :any:`BaseConv2D`.

    with_bias : bool
        When True bias term is added after upsampling.
        This parameter needs not to match the original convolution.

    output_shape : tuple of 4 ints
        The shape of upsampled input. When this is omitted, must give
        `original_input` parameter using `set_parameter_variables` method,
        so that output shape can be inferred at build time. Cannot contain
        `None` when using Tensorflow backend.

    data_format : str
        NCHW or NHWC. When output_shape is given, by supplying this format,
        output_shape is automatically converted to runtime format.

    Notes
    -----
    When ``padding='SAME'``, theano backend and tensorflow backend produces
    slightly different, as internal padding mechanism is different, thus cannot
    be 100% numerically compatible.
    """
    # TODO: Add reuse
    def __init__(
            self, _sentinel=None,
            filter_height=None, filter_width=None, n_filters=None,
            strides=None, padding='VALID',
            initializers=None, with_bias=True,
            output_shape=None, data_format=None):
        if _sentinel is not None:
            raise ValueError(
                'Constructor arguments of Conv2DTranspose must be keywards.'
            )

        super(BaseConv2DTranspose, self).__init__(
            filter_height=filter_height, filter_width=filter_width,
            n_filters=n_filters, strides=strides, padding=padding,
            initializers=initializers or {}, with_bias=with_bias,
            output_shape=output_shape, data_format=data_format)

        # TODO Add switch for filter
        self._create_parameter_slot('filter', train=True, serialize=True)
        self._create_parameter_slot(
            'original_input', train=False, serialize=False)
        if with_bias:
            self._create_parameter_slot('bias', train=True, serialize=True)

    def get_parameter_variables(self, name=None):
        variables = {
            key: value for key, value in self._parameter_variables.items()
            if not key == 'original_input'}
        if name:
            return variables[name]
        return variables.values()
