"""Define common interface of Optimizer"""
from __future__ import absolute_import

import abc
import logging

import luchador.util

__all__ = ['BaseOptimizer', 'get_optimizer']
_LG = logging.getLogger(__name__)


def _remove_dup(grads_and_vars):
    """Remove duplicated Variables from grads_and_vars"""
    # http://stackoverflow.com/a/480227
    seen = set()
    seen_add = seen.add
    return [x for x in grads_and_vars if not (x[1] in seen or seen_add(x[1]))]


def _log_wrt(wrt):
    if not luchador.util.is_iteratable(wrt):
        wrt = [wrt]
    for var in wrt:
        _LG.info('    %20s', var)


class BaseOptimizer(luchador.util.StoreMixin, object):
    """Define common interface of Optimizer"""
    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        super(BaseOptimizer, self).__init__()
        self._store_args(**kwargs)
        self.slot = []

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

    def compute_gradients(self, loss, wrt, **kwargs):
        """Compute gradient of loss with respect to wrt.

        This method works in similar way as Tensorflow Optimizers'
        compute_gradient method.

        Parameters
        ----------
        loss : Tensor
            Tensor holding loss value to be minimized

        wrt : [list of] Tensors
            Variables with which gradients of loss are computed. Variables
            marked as not trainable are ignored.

        kwargs
            [Tensorflow only] Other arguments passed to compute_gradients of
            underlying Tenasorflow native Optimizer.

        Returns
        -------
        list of Tensor pairs
            Gradient and corresponding variable pairs. Each tensor is
            not wrapped with Luchador's Variable but bare TensorVariable
            native to backend.
        """
        _LG.info('Computing gradient for %s', loss)
        _log_wrt(wrt)
        return self._compute_gradients(loss, wrt, **kwargs)

    @abc.abstractmethod
    def _compute_gradients(self, loss, wrt, **kwargs):
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

    def get_parameter_variables(self):
        """Get the list of parameter variables used by optimizers

        Returns
        -------
        OrderedDict
            Keys are the names of parameter variables and value are Tensors
        """
        return self.slot


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
