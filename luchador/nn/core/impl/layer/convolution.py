"""Define common interface for Layer classes"""
from __future__ import division
from __future__ import absolute_import

from ...base import BaseLayer
from ...backend import layer

__all__ = ['Conv2D', 'Conv2DTranspose']
# pylint: disable=abstract-method


class Conv2D(layer.Conv2D, BaseLayer):
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

    with_bias : bool
        When True bias term is added after convolution

    name : str
        Used as base scope when building parameters and output

    kwargs
        use_cudnn_on_gpu : bool
            [Tensorflow only] Argument passed to ``tf.nn.conv2d``

        data_format : str
            [Tensorflow only] Argument passed to ``tf.nn.conv2d``. Use this
            if you need to ignore the default runtime convolution format (
            defined with `LUCHADOR_NN_CONV_FORMAT` environmental variable)
            and to force the format.

    Notes
    -----
    To fetch paramter variables with :any:`get_variable`, use keys
    ``filter`` and ``bias`` in the same scope as layer build.
    """
    def __init__(
            self, filter_height, filter_width, n_filters, strides,
            padding='VALID', initializers=None, with_bias=True,
            name='Conv2D', **kwargs):
        super(Conv2D, self).__init__(
            filter_height=filter_height, filter_width=filter_width,
            n_filters=n_filters, strides=strides, padding=padding,
            initializers=initializers or {}, with_bias=with_bias,
            name=name, **kwargs)

        self._create_parameter_slot(
            'filter', val=None, train=True, serialize=True)
        if with_bias:
            self._create_parameter_slot(
                'bias', val=None, train=True, serialize=True)


class Conv2DTranspose(layer.Conv2DTranspose, BaseLayer):
    """Upsample 2D array with reversed convolution (gradient of convolution)

    Internally (both Tensorflow, Theano), this re-uses the implementation for
    gradient computation of convolution, thus construction of this layer is
    somewhat confusing. You need to feed the constructor parameters for the
    original Conv2D layer, and you need to provide the shape of the original
    input to the Conv2D layer either as constructor argument or by setting
    parameter variable `original_input`. This is because, by nature, depending
    on the configuration, the shape of the output of this layer (the gradient
    of the original convolution layer) may not be determined uniquely.

    Examples
    --------
    Create convolution layer, assuming we are using NCHW data format.

    >>> conv2d = nn.layer.Conv2D(
    >>>     filter_height=7, filter_width=5, n_filters=3,
    >>>     strides=3, padding='valid')
    >>> input = nn.Input(shape=(32, 4, 84, 84), name='original_input')
    >>> with nn.variable_scope('convolution'):
    >>>     conv_output = conv2d(input)
    >>> print(conv_output.shape)
    (32, 3, 26, 27)

    Create convolution transpose layer with the same convolution parameter
    (``filter_height``, ``filter_width``, ``n_filters``, ``strides``,
    ``padding``) as the original convolution, and feed the original input
    shape as ``output_shape``.

    >>> conv2d_t = nn.layer.Conv2DTranspose(
    >>>     filter_height=7, filter_width=5, n_filters=3,
    >>>     strides=3, padding='valid', output_shape=(32, 4, 84, 84))
    >>> with nn.variable_scope('transpose'):
    >>>     conv_t_output = conv2d_t(conv_output)
    >>> print(conv_t_output.shape)
    (32, 4, 84, 84)

    As it is tedious to manually provide ``output_shape``, you can provide
    ``original_input`` parameter separately and omit ``output_shape`` in
    constructor.

    >>> conv2d_t2 = nn.layer.Conv2DTranspose(
    >>>     filter_height=7, filter_width=5, n_filters=3,
    >>>     strides=3, padding='valid')
    >>> conv2d_t2.set_parameter_variables(original_input=input)
    >>> with nn.variable_scope('transpose_2'):
    >>>     conv_t2_output = conv2d_t2(conv_output)
    >>> print(conv_t2_output.shape)
    (32, 4, 84, 84)

    Similarly, you can provide reference to the original ``filter`` parameter
    and omit filter shape parameters in constructor. You however cannot omit
    ``strides`` and ``padding``.

    >>> conv2d_t3 = nn.layer.Conv2DTranspose(strides=3, padding='valid')
    >>> original_filter = conv2d.get_parameter_variables('filter')
    >>> conv2d_t3.set_parameter_variables(
    >>>     original_input=input, original_filter=original_filter)
    >>> with nn.variable_scope('transpose_3'):
    >>>     conv_t3_output = conv2d_t3(conv_output)
    >>> print(conv_t3_output.shape)
    (32, 4, 84, 84)

    If you want to use the same filter parameter as the original convolution
    layer (tied weights), you can feed the parameter as ``filter`` instead of
    ``original_filter``

    >>> conv2d_t4 = nn.layer.Conv2DTranspose(strides=3, padding='valid')
    >>> conv2d_t4.set_parameter_variables(
    >>>     original_input=input, filter=original_filter)
    >>> conv_t4_output = conv2d_t4(conv_output)
    >>> print(conv_t4_output.shape)
    (32, 4, 84, 84)

    :py:func:`luchador.nn.util.model_maker.make_layer` function can handle this
    by adding ``parameters`` field. The following configuration will create the
    same layer as ``conv2d_t4``.

    .. code-block:: YAML

        typename: Conv2DTranspose
        args:
            strides: 3
            padding: valid
            with_bias: True
        parameters:
            filter:
                typename: Variable
                name: convolution/filter
            original_input:
                typename: Input
                reuse: True
                name: input

    If you are providing ``output_shape`` constructor argument in YAML file,
    you cannot know in which convolution format luchador will be run. So by
    adding ``output_shape_format`` which describes which convolution format it
    adopts, the layer convert the ``output_shape`` automatically.

    The following configuration file can be used in both ``NCHW`` and ``NHWC``
    format.

    .. code-block:: YAML

        typename: Conv2DTranspose
        args:
            strides: 3
            padding: valid
            with_bias: True
            output_shape: [32, 32, 20, 20]
            output_shape_format: NCHW
        parameters:
            original_input:
                typename: Input
                reuse: True
                name: input
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
        Not optional. See :any:`Conv2D`. This has to consistent with input
        shape and output shape.

    padding : (str or int or tuple of two ints)
        Not optional. See :any:`Conv2D`. This has to consistent with input
        shape and output shape.

    initializers: dict
        bias : dict
            Bias initializer configurations
        filter : dict
            Filter initializer configurations

    with_bias : bool
        When True bias term is added after upsampling.
        This parameter does not have to match with the original convolution.

    output_shape : tuple of 4 ints
        The shape of upsampled input. When this is omitted, must give
        `original_input` parameter using `set_parameter_variables` method,
        so that output shape can be inferred at build time. Cannot contain
        `None` when using Tensorflow backend.

    output_shape_format : str
        NCHW or NHWC. When output_shape is given, by supplying this format,
        output_shape is automatically converted to runtime format.

    name : str
        Used as base scope when building parameters and output

    kwargs
        data_format : str
            [Tensorflow only] Argument passed to ``tf.nn.conv2d``. Use this
            if you need to ignore the default runtime convolution format (
            defined with `LUCHADOR_NN_CONV_FORMAT` environmental variable)
            and to force the format.

    Notes
    -----
    When ``padding='SAME'``, theano backend and tensorflow backend produces
    slightly different, as internal padding mechanism is different, thus cannot
    be 100% numerically compatible.
    """
    def __init__(
            self, _sentinel=None,
            filter_height=None, filter_width=None, n_filters=None,
            strides=None, padding='VALID', initializers=None,
            with_bias=True, output_shape=None, output_shape_format=None,
            name='Conv2DTranspose', **kwargs):
        super(Conv2DTranspose, self).__init__(
            filter_height=filter_height, filter_width=filter_width,
            n_filters=n_filters, strides=strides, padding=padding,
            initializers=initializers or {}, with_bias=with_bias,
            output_shape=output_shape, output_shape_format=output_shape_format,
            name=name, **kwargs)

        self._create_parameter_slot(
            'filter', val=None, train=True, serialize=True)
        if with_bias:
            self._create_parameter_slot(
                'bias', val=None, train=True, serialize=True)
        self._create_parameter_slot(
            'original_input', val=None, train=False, serialize=False)
        self._create_parameter_slot(
            'original_filter', val=None, train=False, serialize=False)
