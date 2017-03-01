from __future__ import absolute_import

from ..base.initializer import BaseInitializer
from .. backend import initializer as init_module


###############################################################################
# pylint: disable=abstract-method
class Constant(init_module.Constant, BaseInitializer):
    """Initialize Variale with constant value

    Parameters
    ----------
    value : number
        Value to initialize Variable

    dtype : str or None
        Data type to sample. If None, default dtype is used.
    """
    def __init__(self, value, dtype=None):
        super(BaseConstant, self).__init__(value=value, dtype=dtype)


class BaseUniform(BaseInitializer):
    """Initialize Variale with samples from uniform distribution

    Parameters
    ----------
    minval, maxval : float
        Minimum/maximum value of sampling distribution

    seed : int or None
        Seed value for random generator

    dtype : str or None
        Data type to sample. If None, default dtype is used.
    """
    def __init__(self, minval=0.0, maxval=1.0, seed=None, dtype=None):
        super(BaseUniform, self).__init__(
            minval=minval, maxval=maxval, seed=seed, dtype=dtype)


class BaseNormal(BaseInitializer):
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
        super(BaseNormal, self).__init__(
            mean=mean, stddev=stddev, seed=seed, dtype=dtype)


class BaseXavier(BaseInitializer):
    # pylint: disable=anomalous-backslash-in-string
    """Implement Xavier initialization [1]_ in Tensorflow manner [2]_

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
    .. [1] Xavier Glorot and Yoshua Bengio (2010):
           Understanding the difficulty of training deep feedforward neural
           networks. International conference on artificial intelligence and
           statistics.
    .. [2]
           https://www.tensorflow.org/versions/r0.11/api_docs/python/contrib.layers.html#xavier_initializer
    """
    def __init__(self, uniform=True, seed=None, dtype=None):
        super(BaseXavier, self).__init__(
            uniform=uniform, seed=seed, dtype=dtype)


class BaseKaiming(BaseInitializer):
    """Implement Kaiming initialization

    References
    ----------
    .. [1] Kaiming He et al. (2015):
           Delving deep into rectifiers: Surpassing human-level performance on
           imagenet classification. arXiv preprint arXiv:1502.01852.
    """
    def __init__(self, uniform=True, seed=None, dtype=None):
        super(BaseKaiming, self).__init__(
            uniform=uniform, seed=seed, dtype=dtype)

