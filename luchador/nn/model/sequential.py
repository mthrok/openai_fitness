"""Define base network model structure and fetch method"""
from __future__ import absolute_import

import logging

import luchador.nn
from .base_model import BaseModel

__all__ = ['Sequential']

_LG = logging.getLogger(__name__)


class LayerConfig(object):  # pylint: disable=too-few-public-methods
    """Class to hold complementary info for Layer class"""
    def __init__(self, layer, scope, input_=None, output=None):
        self.layer = layer
        self.scope = scope
        self.input = input_
        self.output = output

    def __repr__(self):
        return repr({
            'scope': self.scope,
            'layer': self.layer,
            'input': self.input,
            'output': self.output,
        })


class Sequential(BaseModel):
    """Network architecture which produces single output from single input

    Examples
    --------
    >>> from luchador import nn
    >>> cnn = nn.Sequential()
    >>> cnn.add_layer(nn.Conv2D(8, 8, 4, 4), scope='layer1')
    >>> cnn.add_layer(nn.Conv2D(4, 4, 16, 2), scope='layer2')
    >>> cnn.add_layer(nn.Flatten(), scope='layer3')
    >>> cnn.add_layer(nn.Dense(256), scope='layer4')
    >>> cnn.add_layer(nn.Dense(10), scope='layer5')

    The above defines nerwork with two convolution layers followed by
    two affin transformation layers. You can use ``nn.Input`` to define
    input to the network.

    >>> input_tensor = nn.Input(shape=(32, 4, 84, 84))

    `input_tensor` has shape with batch size 32, 4 channels and 84x84 image
    size. (This assumes NCHW format.)

    By passing input Variable, Sequential model instantiate internal
    parameters and return output from the model.

    >>> output_tensor = cnn(input_tensor)

    You can inspect `output_tensor` to check the shape of the tensor.

    >>> print(output_tensor.get_shape())
    (-1, 10)

    To retrieve parameter variables of layers (if there is), use
    ``get_parameter_variables`` method.

    >>> cnn.get_parameter_variables()
    [{'name': 'layer1/weight', 'shape': (4, 4, 8, 8)},
     {'name': 'layer1/bias',   'shape': (4,)},
     {'name': 'layer2/weight', 'shape': (16, 4, 4, 4)},
     {'name': 'layer2/bias',   'shape': (16,)},
     {'name': 'layer4/weight', 'shape': (1296, 256)},
     {'name': 'layer4/bias',   'shape': (256,)},
     {'name': 'layer5/weight', 'shape': (256, 10)},
     {'name': 'layer5/bias',   'shape': (10,)}]

    To retrieve output tensors from each layers, use
    ``get_output_tensors`` method

    >>> cnn.get_output_tensors()
    [{'name': 'layer1/output', 'shape': (32, 4, 20, 20)},
     {'name': 'layer2/output', 'shape': (32, 16, 9, 9)},
     {'name': 'layer3/output', 'shape': (32, 1296)},
     {'name': 'layer4/output', 'shape': (32, 256)},
     {'name': 'layer5/output', 'shape': (32, 10)}]
    """
    def __init__(self):
        super(Sequential, self).__init__()
        # Layer configurations
        self.layer_configs = []
        # I/O tensors of the model
        self.input = None
        self.output = None

    ###########################################################################
    def add_layer(self, layer, scope=None):
        """Add layer to model

        Parameters
        ----------
        layer : Layer
            Layer instance to add to model

        scope : str
            Variable scope used when instantiating layer. Layers parmeters are
            instantiated under this scope.
        """
        self.layer_configs.append(LayerConfig(
            layer=layer,
            scope=scope,
            input_=None,
            output=None,
        ))
        return self

    ###########################################################################
    # Functions for building actual computation graphs
    def __call__(self, input_tensor):
        """Convenience function to call ``build``"""
        return self.build(input_tensor)

    def build(self, input_tensor):
        """Build the model on top of input tensor.

        Parameters
        ----------
        input_tensor : Tensor
            Input to this model

        Returns
        -------
        Tensor
            Output from the last layer
        """
        tensor = self.input = input_tensor
        for cfg in self.layer_configs:
            cfg.input = tensor
            scope = cfg.scope or luchador.nn.get_variable_scope()
            with luchador.nn.variable_scope(scope):
                tensor = cfg.layer(tensor)
            cfg.output = tensor
        self.output = tensor
        return self.output

    ###########################################################################
    # Functions for retrieving variables, tensors and operations
    def get_parameter_variables(self):
        """Get Variable objects consisting the parameters of this model

        Returns
        -------
        list
            List of Variables from layer parameters
        """
        ret = []
        for cfg in self.layer_configs:
            ret.extend(cfg.layer.parameter_variables.values())
        return ret

    def get_output_tensors(self):
        """Get Tensor objects which represent the output of each layer

        Returns
        -------
        list
            List of Tensors each of which hold output from layer
        """
        return [cfg.output for cfg in self.layer_configs]

    def get_update_operations(self):
        """Get update opretaions from each layer

        Returns
        -------
        list
            List of update operations from each layer
        """
        return [cfg.layer.get_update_operation() for cfg in self.layer_configs]

    ###########################################################################
    def __repr__(self):
        return repr({
            'model_type': self.__class__.__name__,
            'layer_configs': self.layer_configs,
        })
