"""Define base network model structure and fetch method"""
from __future__ import absolute_import

import logging

from .base_model import BaseModel

_LG = logging.getLogger(__name__)


class Sequential(BaseModel):
    """Network architecture which produces single output from single input

    Examples
    --------
    >>> from luchador import nn
    >>> cnn = nn.Sequential()
    >>> cnn.add_layer(nn.Conv2D(8, 8, 4, 4, name='layer1'))
    >>> cnn.add_layer(nn.Conv2D(4, 4, 16, 2, name='layer2'))
    >>> cnn.add_layer(nn.Flatten(name='layer3'))
    >>> cnn.add_layer(nn.Dense(256, name='layer4'))
    >>> cnn.add_layer(nn.Dense(10, name='layer5'))

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

    ``get_parameter_variables`` method retrieves all the internal parameters,
    and this may include some internal reference variables, which are not
    suitable for serializatoin/gradient computation.
    Use ``get_parameters_to_serialize`` and ``get_parameters_to_train``
    for retrieving parameters for this purpose.

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
        self.layers = []

    ###########################################################################
    def add_layer(self, layer):
        """Add layer to model

        Parameters
        ----------
        layer : Layer
            Layer instance to add to model
        """
        self.layers.append(layer)
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
        self.output = self.input = input_tensor
        for layer in self.layers:
            self.output = layer(self.output)
        return self.output

    ###########################################################################
    # Functions for retrieving variables, tensors and operations
    def get_parameter_variables(self):
        """Get parameter Variables

        Returns
        -------
        list
            List of Variables from layer parameters
        """
        ret = []
        for layer in self.layers:
            ret.extend(layer.get_parameter_variables())
        return ret

    def get_parameters_to_train(self):
        """Get parameter Variables to be fet to gradient computation.

        Returns
        -------
        list
            List of Variables from layer parameters
        """
        ret = []
        for layer in self.layers:
            ret.extend(layer.get_parameters_to_train())
        return ret

    def get_parameters_to_serialize(self):
        """Get parameter Variables to be serialized.

        Returns
        -------
        list
            List of Variables from layer parameters
        """
        ret = []
        for layer in self.layers:
            ret.extend(layer.get_parameters_to_serialize())
        return ret

    def get_output_tensors(self):
        """Get Tensor objects which represent the output of each layer

        Returns
        -------
        list
            List of Tensors each of which hold output from layer
        """
        return [layer.output for layer in self.layers]

    def get_update_operations(self):
        """Get update opretaions from each layer

        Returns
        -------
        list
            List of update operations from each layer
        """
        ret = []
        for layer in self.layers:
            update = layer.get_update_operation()
            if update:
                ret.append(update)
        return ret

    ###########################################################################
    def __repr__(self):
        return repr({
            'typename': self.__class__.__name__,
            'layers': self.layers,
        })
