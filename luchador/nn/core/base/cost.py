"""Define common interface for Cost classes"""
from __future__ import absolute_import

import abc
import logging

import luchador.util
from .node import Node
from .scope import variable_scope

__all__ = ['BaseCost', 'fetch_cost']
_LG = logging.getLogger(__name__)


class BaseCost(luchador.util.StoreMixin, Node):
    """Define common interface for cost computation class"""
    __metaclass__ = abc.ABCMeta

    def __init__(self, **args):
        super(BaseCost, self).__init__()
        self._store_args(**args)

    def __call__(self, target, prediction):
        """Convenience method to call `build`"""
        return self.build(target, prediction)

    def build(self, target, prediction):
        """Build cost between target and prediction

        Parameters
        ----------
        target : Tensor
            Tensor holding the correct prediction value.

        prediction : Tensor
            Tensor holding the current prediction value.

        Returns
        -------
        Tensor
            Tensor holding the cost between the given input tensors.
        """
        _LG.info(
            '  Building cost %s between target: %s and prediction: %s',
            type(self).__name__, target, prediction
        )
        self.input = {
            'target': target,
            'prediction': prediction,
        }
        with variable_scope(self.args['name']):
            self.output = self._build(target, prediction)
            return self.output

    @abc.abstractmethod
    def _build(self, target, prediction):
        """Actual build method"""
        raise NotImplementedError(
            '`_build` method is not implemented for {}'.format(self.__class__)
        )


def fetch_cost(name):
    """Get ``Cost`` class by name

    Parameters
    ----------
    name : str
        Type of ``Cost`` to get.

    Returns
    -------
    type
        ``Cost`` type found

    Raises
    ------
    ValueError
        When ``Cost`` class with the given type is not found
    """
    for class_ in luchador.util.fetch_subclasses(BaseCost):
        if class_.__name__ == name:
            return class_
    raise ValueError('Unknown Cost: {}'.format(name))
