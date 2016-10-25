"""Define common interface for Layer classes"""

from __future__ import division
from __future__ import absolute_import

import logging
from collections import OrderedDict

from luchador import common

_LG = logging.getLogger(__name__)

__all__ = [
    'BaseLayer', 'get_layer',
    'BaseDense', 'BaseConv2D',
    'BaseReLU', 'BaseSigmoid', 'BaseSoftmax',
    'BaseTrueDiv', 'BaseFlatten',
    'BaseBatchNormalization',
    'BaseNCHW2NHWC', 'BaseNHWC2NCHW',
]


class BaseLayer(common.SerializeMixin, object):
    """Defines common interface (copy and build) for Layer classes"""
    def __init__(self, **kwargs):
        """Validate and store arguments passed to subclass __init__ method

        As these **args are used to create a copy of instance, arguments which
        cannot be passed to constructor should not be passed. In other way, the
        signature of this constructor must match to the signature of the
        constractor of subclass object being created.
        """
        super(BaseLayer, self).__init__()
        self.store_args(**kwargs)

        self.initializers = OrderedDict()
        self.update_operations = OrderedDict()
        self.parameter_variables = OrderedDict()

    ###########################################################################
    # Setter for learnable parameters
    def _add_update(self, name, operation):
        self.update_operations[name] = operation

    def get_update_operation(self):
        """Get operation which updates Layer parameter

        For layers which require updates other than back propagate
        optimization, Operation returned by this function must be
        fed to Session.run function.

        Currently only BatchNormalization requires such operation.
        """
        return self._get_update_operation()

    def _get_update_operation(self):
        raise NotImplementedError(
            '`get_update_operation` method is not implemented for {}'
            .format(self.__class__)
        )

    ###########################################################################
    # Setter/Getter for learnable parameters
    def _add_parameter(self, name, variable):
        self.parameter_variables[name] = variable

    def _get_parameter(self, name):
        return self.parameter_variables[name]

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
            Tensor which holds the output of build computation
        """
        _LG.debug('    Building %s: %s', type(self).__name__, self.args)
        return self._build(input_tensor)

    def _build(self, input_tensor):
        raise NotImplementedError(
            '`_build` method is not implemented for {}'.format(self.__class__)
        )


def get_layer(name):
    for Class in common.get_subclasses(BaseLayer):
        if Class.__name__ == name:
            return Class
    raise ValueError('Unknown Layer: {}'.format(name))


###############################################################################
class BaseDense(BaseLayer):
    """Apply 2D affine transformation.

    Apply affine transformation to 2D tensor. Activation functions, such as
    ReLU are not applied.

    Input shape
    -----------
    (batch size, #input features)

    Output shape
    ------------
    (batch size, #output features)

    """
    def __init__(self, n_nodes, initializers=None, with_bias=True):
        """Initialize dense layer.

        Parameters
        ----------
        n_nodes : int
            The number of internal neurons.

        initializers : dict or None
            Dictionary containing configuration.

        with_bias : bool
            When True, bias term is added after multiplication.
        """
        super(BaseDense, self).__init__(
            n_nodes=n_nodes, initializers=initializers, with_bias=with_bias)


class BaseConv2D(BaseLayer):
    """Apply 2D convolution.

    Apply convolution to 4D tensor

    Input shape
    -----------
    HCHW format
        (batch size, #input channel, #input height, #input width) when data

    NHWC format (Tensorflow backend only)
        (batch size, #input height, #input width, #input channel)
    """
    def __init__(self, filter_height, filter_width, n_filters, strides,
                 padding='VALID', initializers=None, with_bias=True, **kwargs):
        """Initialize 2D convolution layer.

        Parameters
        ----------
        filter_height : int
            filter height, (#rows in filter)

        filter_weight : int
            filter weight (#columns in filter)

        n_filters : int
            #filters (#output channels)

        strides : (int, tuple of two int, or tuple of four int)
            - When given type is int, the output is subsampled by this factor
              in both width and height direction.
            - When given type is tuple of two int, the output is subsapmled
              `strides[0]` in height direction and `striders[1]` in width
              direction.
            - In Tensorflow, when given type is tuple of four int, it must
              be consistent with the input data format. That is:
              - 'NHWC' (default): (batch, height, width, channel)
              - 'NCHW': (batch, channel, height, width)

        padding :
            - [tensorflow] (str): Either 'SAME' or 'VALID'
            - [theano] (str or int or tuple of two int): See Theano doc

        with_bias : bool
            When True bias term is added after convolution

        kwargs :
            Tensorflow: Arguments passed to tf.nn.conv2d. 'use_cudnn_on_gpu'
        """
        super(BaseConv2D, self).__init__(
            filter_height=filter_height, filter_width=filter_width,
            n_filters=n_filters, strides=strides, padding=padding,
            initializers=initializers, with_bias=with_bias, **kwargs)


###############################################################################
class BaseReLU(BaseLayer):
    """Apply rectified linear activation"""
    def __init__(self):
        super(BaseReLU, self).__init__()


class BaseSigmoid(BaseLayer):
    """Apply Sigmoid activation elementwise"""
    def __init__(self):
        super(BaseSigmoid, self).__init__()


class BaseSoftmax(BaseLayer):
    """Apply Softmax activation"""
    def __init__(self):
        super(BaseSoftmax, self).__init__()


###############################################################################
class BaseTrueDiv(BaseLayer):
    """Apply reald-valued division to input tensor elementwise"""
    def __init__(self, denom, dtype=None):
        super(BaseTrueDiv, self).__init__(denom=denom, dtype=dtype)
        self.denom = None


class BaseFlatten(BaseLayer):
    """Reshape 4D tensor into 2D tensor"""
    def __init__(self):
        super(BaseFlatten, self).__init__()


###############################################################################
class BaseBatchNormalization(BaseLayer):
    """Apply batch normalization [1]_:

    .. math::
        y = \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} \\gamma + \\beta

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
