"""Define GraphComponent class"""
from __future__ import absolute_import

from collections import OrderedDict

import luchador.util

__all__ = ['Node', 'get_node']


class Node(object):  # pylint: disable=too-few-public-methods
    """Make subclass retrievable with get_node method

    This class was introduced to make it possible to incorporate classes such
    as Cost and Optimizer into Model class, so that we can define models
    consisting of not only Layers but other classes from configuration file.
    """
    def __init__(self):
        super(Node, self).__init__()
        self.input = None
        self.output = None

        self._parameter_variables = OrderedDict()
        self._update_operations = []

        self._parameters_to_train = []
        self._parameters_to_serialize = []

    ###########################################################################
    # Getter/Setter for learnable parameters
    def _create_parameter_slot(
            self, name, val=None, train=True, serialize=True):
        self._parameter_variables[name] = val
        if train:
            self._parameters_to_train.append(name)
        if serialize:
            self._parameters_to_serialize.append(name)

    def get_parameter_variable(self, name):
        """Get parameter variables

        Parameters
        ----------
        name : str
            The name of the parameter (such as ``weight``) to retrieve.

        Returns
        -------
        Variable
        """
        return self._parameter_variables[name]

    def get_parameters_to_train(self):
        """Get parameter variables for training.

        This function is mainly for retrieving variables which are fed to
        gradient computation as `wrt`.

        Returns
        -------
        list of Variable
        """
        return [
            self._parameter_variables[key]
            for key in self._parameters_to_train]

    def get_parameters_to_serialize(self):
        """Get parameter variables for serialization.

        This function returns parameter variables need to be serialized.

        Returns
        -------
        list of Variables
        """
        return [
            self._parameter_variables[key]
            for key in self._parameters_to_serialize]

    def set_parameter_variables(self, **variables):
        """Set parameter variables

        Parameters
        ----------
        variables : dict
            Name and Variable pair. See each Node's documentation to find the
            of correct name to give.
        """
        for key in variables:
            if key not in self._parameter_variables:
                raise ValueError(
                    'Unexpected parameter name: `{}`. Accepted names are {}'
                    .format(key, self._parameter_variables.keys()))
        self._parameter_variables.update(variables)

    ###########################################################################
    # Getter for update operation
    def get_update_operations(self):
        """Get Operation which updates Node parameter

        For optimizers, this will return the update operation created with the
        last call to ``minimize`` method.

        For layers which require updates other than back propagate
        optimization, Operation returned by this function must be
        fed to Session.run function. (Currently only BatchNormalization
        requires such operation.)

        Returns
        -------
        Operation or None
            If update Operation is defined (BatchNormalization), Operation is
            returned, else None
        """
        return self._update_operations


def get_node(name):
    """Get ``Node`` class by name

    Parameters
    ----------
    name : str
        Type of ``Node`` to get

    Returns
    -------
    type
        ``Node`` type found

    Raises
    ------
    ValueError
        When ``Node`` class with the given type is not found
    """
    for class_ in luchador.util.get_subclasses(Node):
        if class_.__name__ == name:
            return class_
    raise ValueError('Unknown Node: {}'.format(name))
