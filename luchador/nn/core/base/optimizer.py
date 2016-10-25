from __future__ import absolute_import

from luchador import common

__all__ = [
    'BaseOptimizer', 'get_optimizer',
    'BaseSGD',
    'BaseRMSProp', 'BaseNeonRMSProp', 'BaseGravesRMSProp',
    'BaseAdam', 'BaseAdamax',
]


class BaseOptimizer(common.SerializeMixin):
    """Defines common interface for gradient computation and application"""
    def __init__(self, **kwargs):
        super(BaseOptimizer, self).__init__()
        self.store_args(**kwargs)
        self.slot = []

        # Backend-specific initialization is run here
        self.init()

    def init(self):
        """Backend-specific initilization"""
        raise NotImplementedError(
            '`init` is not implemnted for {}.{}'
            .format(type(self).__module__, type(self).__name__)
        )

    def minimize(self, loss, wrt, **kwargs):
        """Minimize loss with the given variables

        Args:
          loss (TensorWrapper):Loss value to be minimized
          wrt (list of Variables): Variables with which loss is minimzied

        Returns:
          OperationWrapper: Minimization operation
        """
        raise NotImplementedError(
            '`minimize` is not implemnted for {}.{}'
            .format(type(self).__module__, type(self).__name__)
        )

    def compute_gradients(self, loss, wrt, **kwargs):
        """Compute gradient of loss with respect to wrt.

        This method works in similar way as Tensorflow Optimizers'
        compute_gradient method.

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
    for Class in common.get_subclasses(BaseOptimizer):
        if Class.__name__ == name:
            return Class
    raise ValueError('Unknown Optimizer: {}'.format(name))


###############################################################################
class BaseSGD(BaseOptimizer):
    def __init__(self, learning_rate, name='SGD', **kwargs):
        super(BaseSGD, self).__init__(
            learning_rate=learning_rate, name=name, **kwargs)


class BaseRMSProp(BaseOptimizer):
    def __init__(self, learning_rate,
                 decay=0.95, momentum=0.0,
                 epsilon=1e-2, name='RMSProp', **kwargs):
        super(BaseRMSProp, self).__init__(
            learning_rate=learning_rate, decay=decay, momentum=momentum,
            epsilon=epsilon, name=name, **kwargs)


class BaseNeonRMSProp(BaseOptimizer):
    def __init__(self, learning_rate, decay=0.95, epsilon=1e-6,
                 name='NeonRMSProp', **kwargs):
        super(BaseNeonRMSProp, self).__init__(
            learning_rate=learning_rate, decay=decay,
            epsilon=epsilon, name=name, **kwargs)


class BaseGravesRMSProp(BaseOptimizer):
    """RMSProp used in DQN paper[1] and described in A.Graves paper [2]

    [1] https://github.com/kuz/DeepMind-Atari-Deep-Q-Learner/blob/4b9f5a79b03ea0cfc512ed1c11f1b00bc875bc57/dqn/NeuralQLearner.lua#L265  # nopep8
    [2] http://arxiv.org/pdf/1308.0850v5.pdf
    """
    def __init__(self, learning_rate,
                 decay1=0.95, decay2=0.95,
                 epsilon=1e-2, name='GravesRMSProp', **kwargs):
        super(BaseGravesRMSProp, self).__init__(
            learning_rate=learning_rate, decay1=decay1, decay2=decay2,
            epsilon=epsilon, name=name, **kwargs)


class BaseAdam(BaseOptimizer):
    def __init__(self, learning_rate,
                 beta1=0.9, beta2=0.999,
                 epsilon=1e-08, name='Adam', **kwargs):
        super(BaseAdam, self).__init__(
            learning_rate=learning_rate, beta1=beta1, beta2=beta2,
            epsilon=epsilon, name=name, **kwargs)


class BaseAdamax(BaseOptimizer):
    def __init__(self, learning_rate,
                 beta1=0.9, beta2=0.999,
                 epsilon=1e-8, name='Adamax', **kwargs):
        super(BaseAdamax, self).__init__(
            learning_rate=learning_rate, beta1=beta1, beta2=beta2,
            epsilon=epsilon, name=name, **kwargs)
