"""Define common interface of Optimizer"""
from __future__ import absolute_import

import abc
import logging

import luchador.util
from .node import Node

__all__ = ['BaseOptimizer', 'get_optimizer']
_LG = logging.getLogger(__name__)


def _remove_dup(grads_and_vars):
    """Remove duplicated Variables from grads_and_vars"""
    # http://stackoverflow.com/a/480227
    seen = set()
    seen_add = seen.add
    return [x for x in grads_and_vars if not (x[1] in seen or seen_add(x[1]))]


class BaseOptimizer(luchador.util.StoreMixin, Node):
    """Define common interface of Optimizer"""
    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        super(BaseOptimizer, self).__init__()
        self._store_args(**kwargs)

        # Backend-specific initialization is run here
        self._run_backend_specific_init()

    @abc.abstractmethod
    def _run_backend_specific_init(self):
        """Backend-specific initilization"""
        pass

    def minimize(self, loss, wrt, **kwargs):
        """Create operation to minimization loss w.r.t. given variables

        Parameters
        ----------
        loss : Tensor
            Tensor holding loss value to be minimized

        wrt : [list of] Variable
            Variables with which loss is minimzied. Variables marked
            as not trainable are ignored.

        kwargs
            [Tensorflow only] Other arguments passed to either
            compute_gradients or apply_gradients of Tenasorflow native
            Optimizer.

        Returns
        -------
        Operation
            Minimization operation
        """
        return self._minimize(loss, wrt, **kwargs)

    @abc.abstractmethod
    def _minimize(self, loss, wrt, **kwargs):
        pass

    def apply_gradients(self, grads_and_vars, **kwargs):
        """Apply gradients to variables

        Parameters
        ----------
        grads_and_vars : list of Tensor pairs
            Valued returned from compute_gradient method.

        Returns
        -------
        Operation
            Operation which updates parameter variables
        """
        grads_and_vars = [
            (grad.unwrap(), var.unwrap())
            for grad, var in grads_and_vars if grad is not None]
        grads_and_vars = _remove_dup(grads_and_vars)
        return self._apply_gradients(grads_and_vars, **kwargs)

    @abc.abstractmethod
    def _apply_gradients(self, grads_and_vars, **kwargs):
        pass


def get_optimizer(name):
    """Get ``Optimizer`` class by name

    Parameters
    ----------
    name : str
        Type of ``Optimizer`` to get

    Returns
    -------
    type
        ``Optimizer`` type found

    Raises
    ------
    ValueError
        When ``Optimizer`` class with the given type is not found
    """
    for class_ in luchador.util.get_subclasses(BaseOptimizer):
        if class_.__name__ == name:
            return class_
    raise ValueError('Unknown Optimizer: {}'.format(name))
