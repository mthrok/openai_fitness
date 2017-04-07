"""Define gradient operation"""
from __future__ import absolute_import

import logging

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
    return be.ops.compute_gradient(loss, wrt, **kwargs)
