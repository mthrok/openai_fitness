"""Define common interface of Optimizer"""
from __future__ import absolute_import

import abc

import luchador.util


class BaseOptimizer(luchador.util.SerializeMixin):
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

        wrt : [list of] Tensors:
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


def get_optimizer(typename):
    """Retrieve Optimizer class by type

    Parameters
    ----------
    typename : str
        Type of Optimizer to retrieve

    Returns
    -------
    type
        Optimizer type found

    Raises
    ------
    ValueError
        When Optimizer with the given type is not found
    """
    for class_ in luchador.util.get_subclasses(BaseOptimizer):
        if class_.__name__ == typename:
            return class_
    raise ValueError('Unknown Optimizer: {}'.format(typename))


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
        - use_lock : [TF only] passed to underlying TF native optimizer
    """
    def __init__(self, learning_rate, name='SGD', **kwargs):
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
        use_lock : [Tensorflow only] passed to underlying TF native optimizer

    References
    ----------
    .. [1] Tieleman, T. and Hinton, G. (2012):
           Neural Networks for Machine Learning, Lecture 6.5 - rmsprop.
           Coursera. http://www.youtube.com/watch?v=O3sxAc4hxZU (formula @5:20)
    """
    def __init__(self, learning_rate,
                 decay=0.95, momentum=0.0,
                 epsilon=1e-2, name='RMSProp', **kwargs):
        super(BaseRMSProp, self).__init__(
            learning_rate=learning_rate, decay=decay, momentum=momentum,
            epsilon=epsilon, name=name, **kwargs)


class BaseNeonRMSProp(BaseOptimizer):
    """Neon style RMSProp

    The update rule is similar to :any:`BaseRMSProp` without moemntum, but
    epsilon appears twice.

    .. math::
        rms_t &= \\rho * rms_{t-1} + (1- \\rho) * grad ^2 \\\\
        lr_t &= \\frac{lr}{\\sqrt{rms_t + \\epsilon} + \\epsilon} \\\\
        var_t &= var_{t-1} - lr * grad  \\\\

    where :math:`\\rho` is decay ratio

    Parameters
    ----------
    learning_rate : float
        The learning rate controlling the size of update steps
    decay : float
        Decay factor at which rate accumurated RMS decays.
    epsilon : float
        Small value added for numerical stability
    name : str
        Used to create scope which contains parameter variables
    kwargs
        use_lock : [Tensorflow only] passed to underlying TF native optimizer

    References
    ----------
    .. [1] Tieleman, T. and Hinton, G. (2012):
           Neural Networks for Machine Learning, Lecture 6.5 - rmsprop.
           Coursera. http://www.youtube.com/watch?v=O3sxAc4hxZU (formula @5:20)
    """
    def __init__(self, learning_rate, decay=0.95, epsilon=1e-6,
                 name='NeonRMSProp', **kwargs):
        super(BaseNeonRMSProp, self).__init__(
            learning_rate=learning_rate, decay=decay,
            epsilon=epsilon, name=name, **kwargs)


class BaseGravesRMSProp(BaseOptimizer):
    """RMSProp used in DQN paper [1]_ and described in A.Graves paper [2]_

    References
    ----------
    .. [1] Mnih, V et. al (2015)
           Human-level control through deep reinforcement learning
           https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
           https://github.com/kuz/DeepMind-Atari-Deep-Q-Learner/blob/4b9f5a79b03ea0cfc512ed1c11f1b00bc875bc57/dqn/NeuralQLearner.lua#L265
    .. [2] Graves, A. (2014):
           Generating Sequences With Recurrent Neural Networks
           http://arxiv.org/pdf/1308.0850v5.pdf
    """
    def __init__(self, learning_rate,
                 decay1=0.95, decay2=0.95,
                 epsilon=1e-2, name='GravesRMSProp', **kwargs):
        super(BaseGravesRMSProp, self).__init__(
            learning_rate=learning_rate, decay1=decay1, decay2=decay2,
            epsilon=epsilon, name=name, **kwargs)


class BaseAdam(BaseOptimizer):
    """Adam optimizer [1]_

    References
    ----------
    .. [1] Kingma, D. Ba, J 2014
        Adam: A Method for Stochastic Optimization
        https://arxiv.org/abs/1412.6980
    """
    def __init__(self, learning_rate,
                 beta1=0.9, beta2=0.999,
                 epsilon=1e-08, name='Adam', **kwargs):
        super(BaseAdam, self).__init__(
            learning_rate=learning_rate, beta1=beta1, beta2=beta2,
            epsilon=epsilon, name=name, **kwargs)


class BaseAdamax(BaseOptimizer):
    """Adam optimizer [1]_

    References
    ----------
    .. [1] Kingma, D. Ba, J 2014
        Adam: A Method for Stochastic Optimization
        https://arxiv.org/abs/1412.6980
    """
    def __init__(self, learning_rate,
                 beta1=0.9, beta2=0.999,
                 epsilon=1e-8, name='Adamax', **kwargs):
        super(BaseAdamax, self).__init__(
            learning_rate=learning_rate, beta1=beta1, beta2=beta2,
            epsilon=epsilon, name=name, **kwargs)
