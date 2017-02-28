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


###############################################################################
# pylint: disable=abstract-method
class BaseSGD(BaseOptimizer):
    """Implement Stochastic Gradient Descent

    Parameters
    ----------
    learning_rate : float
        The learning rate controlling the size of update steps
    name : str
        Used to create scope which contains parameter variables.
        Virtually has no effect in SGD
    kwargs
        Other accepted keyword arguments
        use_lock
            [Tensorflow nly] passed to underlying TF native optimizer
    """
    def __init__(
            self, learning_rate, name='SGD', **kwargs):
        super(BaseSGD, self).__init__(
            learning_rate=learning_rate, name=name, **kwargs)


class BaseRMSProp(BaseOptimizer):
    """Tensorflow style RMSProp with momentum

    Scale learning rates by dividing with the moving average of the root mean
    squared (RMS) gradients. See [1]_ for further description.

    This implementation mimics TF native RMSProp, which updates parameters as

    .. math::
        rms_t &= \\rho * rms_{t-1} + (1- \\rho) * grad ^2 \\\\
        lr_t &= \\frac{lr}{\\sqrt{rms_t + \\epsilon}} \\\\
        mom_t &= \\gamma * mom_{t-1} + lr * grad  \\\\
        var_t &= var_{t-1} - mom_t

    where :math:`\\rho` is decay ratio and :math:`\\gamma` is momentum
    coefficient.

    Parameters
    ----------
    learning_rate : float
        The learning rate controlling the size of update steps
    decay : float
        Decay factor at which rate accumurated RMS decays.
    momentum : float
        Momentum coefficient at which rate parameter update is accumurated.
    epsilon : float
        Small value added for numerical stability
    name : str
        Used to create scope which contains parameter variables
    kwargs
        Other accepted keyword arguments
        use_lock
            [Tensorflow nly] passed to underlying TF native optimizer

    References
    ----------
    .. [1] Tieleman, T. and Hinton, G. (2012):
           Neural Networks for Machine Learning, Lecture 6.5 - rmsprop.
           Coursera. http://www.youtube.com/watch?v=O3sxAc4hxZU (formula @5:20)
    """
    def __init__(
            self, learning_rate, decay=0.95, momentum=0.0,
            epsilon=1e-2, name='RMSProp', **kwargs):
        super(BaseRMSProp, self).__init__(
            learning_rate=learning_rate, decay=decay,
            momentum=momentum, epsilon=epsilon, name=name, **kwargs)


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
