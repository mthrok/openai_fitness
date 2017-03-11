"""Define base network model structure and fetch method"""
from __future__ import absolute_import

import logging

from .graph import Graph

_LG = logging.getLogger(__name__)


class Sequential(Graph):
    """Network architecture which produces single output from single input

    This class is a specialization of :any:``Graph`` model, where you only
    need to supply one input to the model and the resulting output is
    propageted layer by layer. As a result, you cannot use nodes that require
    multiple inputs such as Cost.

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
    def __init__(self, name=None):
        super(Sequential, self).__init__(name=name)

    @property
    def layers(self):
        """Layer lists"""
        return self.nodes

    @property
    def input(self):
        """Return the input of the first layer else None"""
        if self.layers:
            return self.layers[0].input
        return None

    @input.setter
    def input(self, val):
        """For resolving inheritance conflict"""
        pass

    @property
    def output(self):
        """Return the output of the last layer else None"""
        if self.layers:
            return self.layers[-1].output
        return None

    @output.setter
    def output(self, val):
        """For resolving inheritance conflict"""
        pass

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

    def build(self, tensor):
        """Build the model on top of input tensor.

        Parameters
        ----------
        tensor : Tensor
            Input to this model

        Returns
        -------
        Tensor
            Output from the last layer
        """
        for layer in self.layers:
            tensor = layer(tensor)
        return tensor
