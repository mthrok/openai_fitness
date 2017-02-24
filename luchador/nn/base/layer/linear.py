"""Define common interface for Layer classes"""
from __future__ import division
from __future__ import absolute_import

import logging

from .base import BaseLayer

__all__ = ['BaseDense']

_LG = logging.getLogger(__name__)

# pylint: disable=abstract-method


class BaseDense(BaseLayer):
    """Apply 2D affine transformation.

    Input Tensor
        2D Tensor with shape (batch size, #input features)

    Output Tensor
        2D Tensor with shape (batch size, #output features)

    Parameters
    ----------
    n_nodes : int
        The number of features after tarnsformation.

    initializers : dict or None
        Dictionary containing configuration.

    with_bias : bool
        When True, bias term is added after multiplication.

    Notes
    -----
    To fetch paramter variables with :any:`get_variable`, use keys
    ``weight`` and ``bias`` in the same scope as layer build.
    """
    def __init__(self, n_nodes, initializers=None, with_bias=True):
        super(BaseDense, self).__init__(
            n_nodes=n_nodes, initializers=initializers or {},
            with_bias=with_bias)
