"""Define common interface for Layer classes"""
from __future__ import division
from __future__ import absolute_import

import abc
import logging
from collections import OrderedDict

import luchador.util

_LG = logging.getLogger(__name__)

__all__ = ['BaseLayer']


class BaseLayer(luchador.util.StoreMixin, object):
    """Define common interface (``build``, ``parameters`` ...) of Layer"""
    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        super(BaseLayer, self).__init__()
        self._store_args(**kwargs)

        self._update_operation = None
        self._parameter_variables = OrderedDict()

        self._parameters_to_train = []
        self._parameters_to_serialize = []

    ###########################################################################
    # Getter/Setter for learnable parameters
    def _create_parameter_slot(self, name, train=True, serialize=True):
        self._parameter_variables[name] = None
        if train:
            self._parameters_to_train.append(name)
        if serialize:
            self._parameters_to_serialize.append(name)

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

    def get_parameters_to_train(self, name=None):
        """Get parameter variables for training.

        This function is mainly for retrieving variables which are fed to
        gradient computation as `wrt`.

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
        if name and name not in self._parameters_to_train:
            raise ValueError(
                'Unexpected training parameter name ({}) was given. '
                'Must be one of {}'.format(name, self._parameters_to_train)
            )
        if name:
            return self._parameter_variables[name]
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
            Name and Variable pair. See each Layer's documentation to find the
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
