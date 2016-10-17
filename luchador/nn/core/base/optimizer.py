from __future__ import absolute_import

from luchador.common import get_subclasses, SerializeMixin

__all__ = ['Optimizer', 'get_optimizer', 'make_optimizer']


class Optimizer(SerializeMixin):
    """Defines common interface for gradient computation and application"""
    def __init__(self, **kwargs):
        super(Optimizer, self).__init__()
        self._store_args(**kwargs)
        self.slot = []

    def minimize(self, loss, wrt, **kwargs):
        """Minimize loss with the given variables

        Args:
          loss (TensorWrapper):Loss value to be minimized
          wrt (list of TensorWrappers): Variables with which loss is minimzied

        Returns:
          OperationWrapper: Minimization operation
        """
        raise NotImplementedError(
            '`minimize` is not implemnted for {}.{}'
            .format(type(self).__module__, type(self).__name__)
        )

    def compute_gradients(self, loss, wrt, **kwargs):
        """Compute gradient of loss with respect to wrt.

        This is similar to Tensorflow's Optimizer.compute_gradient method.

        Args:
          loss (TensorWrapper): Loss value to compute gradients
          wrt (list of TensorWrapper):
            Variables with which gradients are computed

        Returns:
          List of Tensor pairs: Gradient and corresponding variable pairs.
            Each tensor is underlying object and not wrapped with
            TensorWrapper class.
        """
        raise NotImplementedError(
            '`compute_gradients` is not implemnted for {}.{}'
            .format(type(self).__module__, type(self).__name__)
        )

    def apply_gradients(self, grads_and_vars, **kwargs):
        """Apply gradients to variables

        Args:
          grads_and_vars (list of Tensor pairs):
            Return value of compute_gradient method.

        Returns:
          OperationWrapper: Operation which updates variables.
        """
        raise NotImplementedError(
            '`apply_gradients` is not implemnted for {}.{}'
            .format(type(self).__module__, type(self).__name__)
        )

    def get_parameter_variables(self):
        """Get the list of parameter variables used by optimizers"""
        return self.slot


def get_optimizer(name):
    """Get optimizer class"""
    for Class in get_subclasses(Optimizer):
        if Class.__name__ == name:
            return Class
    raise ValueError('Unknown Optimizer: {}'.format(name))


def make_optimizer(cfg):
    Optimizer = get_optimizer(cfg['name'])
    return Optimizer(**cfg['args'])
