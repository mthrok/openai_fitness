"""Define common interface of Optimizer"""
from __future__ import absolute_import

import logging

import luchador.util
from ..base.optimizer import BaseOptimizer
from ..backend import optimizer

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


class SGD(optimizer.SGD, BaseOptimizer):
    """Implement Stochastic Gradient Descent

    Parameters
    ----------
    learning_rate : float
        The learning rate controlling the size of update steps
    scope : str
        Used to create scope which contains parameter variables.
        Virtually has no effect in SGD
    kwargs
        Other accepted keyword arguments
        use_lock
            [Tensorflow nly] passed to underlying TF native optimizer
    """
    def __init__(
            self, learning_rate, scope='SGD', **kwargs):
        super(SGD, self).__init__(
            learning_rate=learning_rate, scope=scope, **kwargs)


class RMSProp(optimizer.RMSProp, BaseOptimizer):
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
    scope : str
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
            epsilon=1e-2, scope='RMSProp', **kwargs):
        super(RMSProp, self).__init__(
            learning_rate=learning_rate, decay=decay,
            momentum=momentum, epsilon=epsilon, scope=scope, **kwargs)


class NeonRMSProp(optimizer.NeonRMSProp, BaseOptimizer):
    """Neon style RMSProp

    The update rule is similar to :any:`RMSProp` without moemntum, but
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
    scope : str
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
            self, learning_rate, decay=0.95, epsilon=1e-6,
            scope='NeonRMSProp', **kwargs):
        super(NeonRMSProp, self).__init__(
            learning_rate=learning_rate,
            decay=decay, epsilon=epsilon, scope=scope, **kwargs)


class GravesRMSProp(optimizer.GravesRMSProp, BaseOptimizer):
    """RMSProp used in DQN paper [1]_ and described in A.Graves paper [2]_

    # TODO: Add docstring
    # TODO: Fix citatoin ref

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
    def __init__(
            self, learning_rate, decay1=0.95, decay2=0.95, epsilon=1e-2,
            scope='GravesRMSProp', **kwargs):
        super(GravesRMSProp, self).__init__(
            learning_rate=learning_rate, decay1=decay1, decay2=decay2,
            epsilon=epsilon, scope=scope, **kwargs)


class Adam(optimizer.Adam, BaseOptimizer):
    """Adam optimizer [1]_

    # TODO: Add docstring
    # TODO: Fix citatoin ref

    References
    ----------
    .. [1] Kingma, D. Ba, J 2014
        Adam: A Method for Stochastic Optimization
        https://arxiv.org/abs/1412.6980
    """
    def __init__(
            self, learning_rate, beta1=0.9, beta2=0.999,
            epsilon=1e-08, scope='Adam', **kwargs):
        super(Adam, self).__init__(
            learning_rate=learning_rate, beta1=beta1, beta2=beta2,
            epsilon=epsilon, scope=scope, **kwargs)


class Adamax(optimizer.Adamax, BaseOptimizer):
    """Adam optimizer [1]_

    # TODO: Add docstring
    # TODO: Fix citatoin ref

    References
    ----------
    .. [1] Kingma, D. Ba, J 2014
        Adam: A Method for Stochastic Optimization
        https://arxiv.org/abs/1412.6980
    """
    def __init__(
            self, learning_rate, beta1=0.9, beta2=0.999,
            epsilon=1e-8, scope='Adamax', **kwargs):
        super(Adamax, self).__init__(
            learning_rate=learning_rate, beta1=beta1, beta2=beta2,
            epsilon=epsilon, scope=scope, **kwargs)
