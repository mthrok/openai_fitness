from __future__ import division
from __future__ import absolute_import

import logging
from collections import OrderedDict

from luchador.common import get_subclasses, SerializeMixin

_LG = logging.getLogger(__name__)

__all__ = ['Layer', 'get_layer']


class Layer(SerializeMixin, object):
    """Defines common interface (copy and build) for Layer classes"""
    def __init__(self, **kwargs):
        """Validate and store arguments passed to subclass __init__ method

        As these **args are used to create a copy of instance, arguments which
        cannot be passed to constructor should not be passed. In other way, the
        signature of this constructor must match to the signature of the
        constractor of subclass object being created.
        """
        super(Layer, self).__init__()
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
    for Class in get_subclasses(Layer):
        if Class.__name__ == name:
            return Class
    raise ValueError('Unknown Layer: {}'.format(name))
