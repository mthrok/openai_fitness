"""Define gradient operation"""
from __future__ import absolute_import

import logging

from luchador.util import is_iteratable
from ... import backend as be

__all__ = ['compute_gradient']
_LG = logging.getLogger(__name__)


def compute_gradient(loss, wrt, **kwargs):
    """Compute gradient

    Parameters
    ----------
    loss : Tensor
        loss to be minimized

    wrt : Variable or list of Variables
        Term for which loss Tensor is differentiated.

    kwargs
        Other arguments passed to ``theano.gradient.grad``

    Returns
    -------
    list
        List of (gradient, variable) pairs
    """
    _LG.info('Computing gradient for %s', loss)

    wrt = wrt if is_iteratable(wrt) else [wrt]
    for var in wrt:
        _LG.info('    %20s', var)
    wrt_ = [v.unwrap() for v in wrt if v.trainable]

    if not wrt_:
        raise ValueError('No variables to optimize.')

    return be.ops.compute_gradient(loss, wrt, **kwargs)
