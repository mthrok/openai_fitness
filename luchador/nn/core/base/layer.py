from __future__ import division
from __future__ import absolute_import

import logging
from collections import OrderedDict

from luchador.common import get_subclasses, SerializeMixin

_LG = logging.getLogger(__name__)

__all__ = [
    'BaseLayer', 'get_layer',
    'BaseDense', 'BaseConv2D', 'BaseReLU', 'BaseFlatten', 'BaseTrueDiv',
    'BaseBatchNormalization',
]


class BaseLayer(SerializeMixin, object):
    """Defines common interface (copy and build) for Layer classes"""
    def __init__(self, **kwargs):
        """Validate and store arguments passed to subclass __init__ method

        As these **args are used to create a copy of instance, arguments which
        cannot be passed to constructor should not be passed. In other way, the
        signature of this constructor must match to the signature of the
        constractor of subclass object being created.
        """
        super(BaseLayer, self).__init__()
        self._store_args(**kwargs)

        self.initializers = OrderedDict()
        self.update_operations = OrderedDict()
        self.parameter_variables = OrderedDict()

    ###########################################################################
    # Setter for learnable parameters
    def _add_update(self, name, op):
        self.update_operations[name] = op

    def get_update_operations(self):
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
        """Build layer computation graph on top of the given tensor"""
        return self.build(input_tensor)

    def build(self, input_tensor):
        """Build layer computation graph on top of the given tensor"""
        raise NotImplementedError(
            '`build` method is not implemented for {}'.format(self.__class__)
        )


def get_layer(name):
    for Class in get_subclasses(BaseLayer):
        if Class.__name__ == name:
            return Class
    raise ValueError('Unknown Layer: {}'.format(name))


###############################################################################
class BaseDense(BaseLayer):
    def __init__(self, n_nodes, initializers={}, with_bias=True):
        """Initialize dense layer.
        Also called fully connected, affine, linear or inner product.

        Activation function, such as ReLU is not included.

        Args:
          n_nodes (int): The number of internal neurons.
          initializers (dict): Dictionary containing configuration.
          with_bias (bool): When True bias term is added after multiplication
        """
        super(BaseDense, self).__init__(
            n_nodes=n_nodes, initializers=initializers, with_bias=with_bias)


class BaseConv2D(BaseLayer):
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

          with_bias (bool): When True bias term is added after convolution

          kwargs:
            - Tensorflow: Arguments passed to tf.nn.conv2d.
              'use_cudnn_on_gpu' and 'name'
        """
        super(BaseConv2D, self).__init__(
            filter_height=filter_height, filter_width=filter_width,
            n_filters=n_filters, strides=strides, padding=padding,
            initializers=initializers, with_bias=with_bias, **kwargs)


class BaseReLU(BaseLayer):
    """Applies Rectified Linear Unit"""
    def __init__(self):
        super(BaseReLU, self).__init__()


class BaseFlatten(BaseLayer):
    """Reshape batch into 2D (batch_size, n_features) from 4D"""
    def __init__(self):
        super(BaseFlatten, self).__init__()


class BaseTrueDiv(BaseLayer):
    """Applies element wise division"""
    def __init__(self, denom, dtype=None):
        super(BaseTrueDiv, self).__init__(denom=denom, dtype=None)
        self.denom = None


class BaseBatchNormalization(BaseLayer):
    """Applies batch normalization

    Ioffe, Sergey and Szegedy, Christian (2015):
           Batch Normalization: Accelerating Deep Network Training by Reducing
           Internal Covariate Shift. http://arxiv.org/abs/1502.03167.
    """
    def __init__(self, scale=1.0, center=0.0, epsilon=1e-4,
                 learn=True, decay=0.999):
        super(BaseBatchNormalization, self).__init__(
            decay=decay, epsilon=epsilon,
            scale=scale, center=center, learn=learn)
