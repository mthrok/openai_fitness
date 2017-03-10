"""Define common interface for Layer classes"""
from __future__ import division
from __future__ import absolute_import

from ...base import BaseLayer
from ...backend import layer

__all__ = ['Dense']
# pylint: disable=abstract-method


class Dense(layer.Dense, BaseLayer):
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

    name : str
        Used as base scope when building parameters and output

    Notes
    -----
    To fetch paramter variables with :any:`get_variable`, use keys
    ``weight`` and ``bias`` in the same scope as layer build.
    """
    def __init__(
            self, n_nodes, initializers=None, with_bias=True, name='Dense'):
        super(Dense, self).__init__(
            n_nodes=n_nodes, initializers=initializers or {},
            with_bias=with_bias, name=name)

        self._create_parameter_slot(
            'weight', val=None, train=True, serialize=True)
        if with_bias:
            self._create_parameter_slot(
                'bias', val=None, train=True, serialize=True)
