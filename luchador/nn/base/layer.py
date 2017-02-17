"""Define common interface for Layer classes"""
from __future__ import division
from __future__ import absolute_import

import abc
import logging
from collections import OrderedDict

import luchador.util

_LG = logging.getLogger(__name__)


class BaseLayer(luchador.util.StoreMixin, object):
    """Define common interface (``build``, ``parameters`` ...) of Layer"""
    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        super(BaseLayer, self).__init__()
        self._store_args(**kwargs)

        self._update_operation = None
        self._parameter_variables = OrderedDict()

    ###########################################################################
    # Getter for learnable parameters
    def get_parameter_variables(self, name=None):
        """Get parameter variables

        Parameters
        ----------
        name : str or None
            The name of the parameter (such as ``weight``) to retrieve.
            If not given, all parameter Variables consisting this layer are
            returned.

        Returns
        -------
        [list of] Variable
            When name is given, a single Variable is returned, otherwise
            list of Variables are returned.
        """
        if name:
            return self._parameter_variables[name]
        return self._parameter_variables.values()

    def get_update_operation(self):
        """Get Operation which updates Layer parameter

        For layers which require updates other than back propagate
        optimization, Operation returned by this function must be
        fed to Session.run function.

        Currently only BatchNormalization requires such operation.

        Returns
        -------
        Operation or None
            If update Operation is defined (BatchNormalization), Operation is
            returned, else None
        """
        return self._update_operation

    ###########################################################################
    # Setter for learnable parameters
    def _add_parameter(self, name, variable):
        self._parameter_variables[name] = variable

    def _get_parameter(self, name):
        return self._parameter_variables[name]

    ###########################################################################
    # Functions for building computation graph
    def __call__(self, input_tensor):
        """Convenience method to call `build`"""
        return self.build(input_tensor)

    def build(self, input_tensor):
        """Build layer computation graph on top of the given tensor

        Parameters
        ----------
        input_tensor : Tensor
            Tensor object. Requirement for this object (such as shape and
            dtype) varies from Layer types.

        Returns
        -------
        Tensor
            Tensor which holds the output of built computation
        """
        _LG.info(
            '  Building layer %s on %s', type(self).__name__, input_tensor)
        return self._build(input_tensor)

    @abc.abstractmethod
    def _build(self, input_tensor):
        raise NotImplementedError(
            '`_build` method is not implemented for {}'.format(self.__class__)
        )


###############################################################################
# pylint: disable=abstract-method
class BaseDense(BaseLayer):
    """Apply 2D affine transformation.

    Input Tensor
        2D Tensor with shape (batch size, #input features)

    Output Tensor
        2D Tensor with shape (batch size, #output features)

    Parameters
    ----------
    n_nodes : int
        The number of features after tarnsformation.

    initializers : dict or None
        Dictionary containing configuration.

    with_bias : bool
        When True, bias term is added after multiplication.

    Notes
    -----
    To fetch paramter variables with :any:`get_variable`, use keys
    ``weight`` and ``bias`` in the same scope as layer build.
    """
    def __init__(self, n_nodes, initializers=None, with_bias=True):
        super(BaseDense, self).__init__(
            n_nodes=n_nodes, initializers=initializers, with_bias=with_bias)


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

    filter_weight : int
        filter weight (#columns in filter)

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
    ``weight`` and ``bias`` in the same scope as layer build.
    """
    def __init__(self, filter_height, filter_width, n_filters, strides,
                 padding='VALID', initializers=None, with_bias=True, **kwargs):
        super(BaseConv2D, self).__init__(
            filter_height=filter_height, filter_width=filter_width,
            n_filters=n_filters, strides=strides, padding=padding,
            initializers=initializers, with_bias=with_bias, **kwargs)


###############################################################################
class BaseReLU(BaseLayer):
    """Apply rectified linear activation elementwise"""
    def __init__(self):
        super(BaseReLU, self).__init__()


class BaseSigmoid(BaseLayer):
    """Apply sigmoid activation elementwise"""
    def __init__(self):
        super(BaseSigmoid, self).__init__()


class BaseTanh(BaseLayer):
    """Apply tanh activation elementwise"""
    def __init__(self):
        super(BaseTanh, self).__init__()


class BaseSin(BaseLayer):
    """Apply sin activation elementwise"""
    def __init__(self):
        super(BaseSin, self).__init__()


class BaseCos(BaseLayer):
    """Apply cos activation elementwise"""
    def __init__(self):
        super(BaseCos, self).__init__()


class BaseSoftmax(BaseLayer):
    """Apply softmax activation elementwise"""
    def __init__(self):
        super(BaseSoftmax, self).__init__()


class BaseSoftplus(BaseLayer):
    """Apply softplus activation elementwise"""
    def __init__(self):
        super(BaseSoftplus, self).__init__()


###############################################################################
class BaseTrueDiv(BaseLayer):
    """Apply real-valued division to input tensor elementwise

    Parameters
    ----------
    denom : float
        The value of denominator
    """
    def __init__(self, denom):
        super(BaseTrueDiv, self).__init__(denom=denom)
        self.denom = None


class BaseMean(BaseLayer):
    """Apply mean to input tensor

    Parameters
    ----------
    axis : int or list of int
        Axis or axes along which to compute the mean
    keep_dim : bool
        If true, retains reduced dimensions with length 1.
    dtype : str
        Output dtype
    """
    def __init__(self, axis, keep_dims=False):
        super(BaseMean, self).__init__(axis=axis, keep_dims=keep_dims)


###############################################################################
class BaseFlatten(BaseLayer):
    """Reshape 4D tensor into 2D tensor"""
    def __init__(self):
        super(BaseFlatten, self).__init__()


class BaseTile(BaseLayer):
    """Tile tensor"""
    def __init__(self, pattern):
        super(BaseTile, self).__init__(pattern=pattern)


###############################################################################
class BaseConcat(BaseLayer):
    """Concatenate variables"""
    def __init__(self, axis=1):
        super(BaseConcat, self).__init__(axis=axis)


class BaseAdd(BaseLayer):
    """Add tensors"""
    def __init__(self):
        super(BaseAdd, self).__init__()


class BaseSub(BaseLayer):
    """Subtract tensors"""
    def __init__(self):
        super(BaseSub, self).__init__()


###############################################################################
class BaseBatchNormalization(BaseLayer):
    """Apply batch normalization [1]_:

    .. math::
        y = \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} \\gamma + \\beta

    Notes
    -----
    To fetch paramter variables with :any:`get_variable`, use keys ``mean``,
    ``var``, ``scale`` and ``offset`` in the same scope as layer build.

    To fetch update operation with :any:`get_operation` use key ``bn_update``
    in the same scope as layer build.

    References
    ----------
    .. [1] Ioffe, Sergey and Szegedy, Christian (2015):
           Batch Normalization: Accelerating Deep Network Training by Reducing
           Internal Covariate Shift. http://arxiv.org/abs/1502.03167.

    """
    def __init__(self, scale=1.0, offset=0.0, epsilon=1e-4,
                 learn=True, decay=0.999):
        super(BaseBatchNormalization, self).__init__(
            decay=decay, epsilon=epsilon,
            scale=scale, offset=offset, learn=learn)

        self._axes = self._pattern = None


###############################################################################
class BaseNHWC2NCHW(BaseLayer):
    """Convert NCHW data from to NHWC

    Rearrange the order of axes in 4D tensor from NHWC to NCHW
    """
    def __init__(self):
        super(BaseNHWC2NCHW, self).__init__()


class BaseNCHW2NHWC(BaseLayer):
    """Convert NCHW data flow to NHWC

    Rearrange the order of axes in 4D tensor from NCHW to NHWC
    """
    def __init__(self):
        super(BaseNCHW2NHWC, self).__init__()
