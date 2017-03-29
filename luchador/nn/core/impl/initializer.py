"""Implement initializers"""
from __future__ import absolute_import

from ..base.initializer import BaseInitializer
from ..backend import initializer

# pylint: disable=anomalous-backslash-in-string


class ConstantInitializer(initializer.Constant, BaseInitializer):
    """Initialize Variale with constant value

    Parameters
    ----------
    value : number
        Value to initialize Variable

    dtype : str or None
        Data type to sample. If None, default dtype is used.
    """
    def __init__(self, value, dtype=None):
        super(ConstantInitializer, self).__init__(value=value, dtype=dtype)


class UniformInitializer(initializer.Uniform, BaseInitializer):
    """Initialize Variale with samples from uniform distribution

    Parameters
    ----------
    min_value, max_value : float
        Minimum/maximum value of sampling distribution

    seed : int or None
        Seed value for random generator

    dtype : str or None
        Data type to sample. If None, default dtype is used.
    """
    def __init__(self, min_value=0.0, max_value=1.0, seed=None, dtype=None):
        if max_value < min_value:
            raise ValueError('`max_value` must be larger than `min_value`')
        super(UniformInitializer, self).__init__(
            min_value=min_value, max_value=max_value, seed=seed, dtype=dtype)


class NormalInitializer(initializer.Normal, BaseInitializer):
    """Initialize Variale with samples from normal distribution

    Parameters
    ----------
    mean, stddev : float
        Mean and standard dedviation of sampling distribution

    seed : int or None
        Seed value for random generator

    dtype : str or None
        Data type to sample. If None, default dtype is used.
    """
    def __init__(self, mean=0.0, stddev=1.0, seed=None, dtype=None):
        super(NormalInitializer, self).__init__(
            mean=mean, stddev=stddev, seed=seed, dtype=dtype)


class XavierInitializer(initializer.Xavier, BaseInitializer):
    """Implement Xavier initialization [AISTATS2010]_ in TF manner [TFXAVIER]_

    Parameters
    ----------
    uniform : Bool
        If True, uniform distribution is used. Otherwise normal distribution
        is used. See Notes.

    seed : int or None
        Seed value for random generator

    dtype : str or None
        Data type to sample. If None, default dtype is used.


    Notes
    -----
    The implementation of Xavier in Theano follows that of Tensorflow.[2]_
    The underlying distribution is computed in the following way.

    when ``uniform=True``,

    .. math::
       a &= \\sqrt{\\frac{6}{fan_{in}+fan_{out}}}\\\\
       W &\sim U[-a, a]

    When ``uniform=False``,

    .. math::
       \\sigma &= \\sqrt{\\frac{3}{fan_{in}+fan_{out}}}\\\\
       W &\sim N(0, \\sigma)

    References
    ----------
    .. [AISTATS2010] Xavier Glorot and Yoshua Bengio (2010):
        Understanding the difficulty of training deep feedforward neural
        networks. International conference on artificial intelligence and
        statistics.
    .. [TFXAVIER]
        https://www.tensorflow.org/versions/r1.0/api_docs/python/tf/contrib/layers/xavier_initializer
    """
    def __init__(self, uniform=True, seed=None, dtype=None):
        super(XavierInitializer, self).__init__(
            uniform=uniform, seed=seed, dtype=dtype)


class KaimingInitializer(initializer.Kaiming, BaseInitializer):
    """Implement Kaiming initialization

    References
    ----------
    .. [ARXIV01852] Kaiming He et al. (2015):
        Delving deep into rectifiers: Surpassing human-level performance on
        imagenet classification. arXiv preprint arXiv:1502.01852.
    """
    def __init__(self, uniform=True, seed=None, dtype=None):
        super(KaimingInitializer, self).__init__(
            uniform=uniform, seed=seed, dtype=dtype)
