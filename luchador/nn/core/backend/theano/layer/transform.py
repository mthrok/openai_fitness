"""Implement Layer classes in Theano"""
from __future__ import division
from __future__ import absolute_import

import logging

import theano.tensor as T

from ..wrapper import Tensor

__all__ = ['Flatten', 'Tile', 'Concat']
_LG = logging.getLogger(__name__)
# pylint:disable=no-member,no-self-use


class Flatten(object):
    """Implement Flatten layer in Theano

    See :any:`BaseFlatten` for detail.
    """
    def _build(self, input_tensor):
        input_shape = input_tensor.shape
        n_nodes = int(reduce(lambda r, d: r*d, input_shape[1:], 1))

        _LG.debug('    Input shape: %s', input_shape)
        _LG.debug('    #Nodes     : %s', n_nodes)

        output_shape = (input_shape[0] or -1, n_nodes)
        output_tensor = T.reshape(input_tensor.unwrap(), output_shape)
        _LG.debug('    output_shape: %s', output_shape)
        return Tensor(output_tensor, shape=output_shape, name='output')


class Tile(object):
    """Implement Tile layer in Theano

    See :any:`BaseFlatten` for detail.
    """
    def _build(self, input_tensor):
        return input_tensor.tile(self.args['pattern'], name='output')


def _compute_concat_shape(shapes, axis):
    _shape = [None] * len(shapes[0])
    _shape[axis] = 0
    for shape in shapes:
        for i, val in enumerate(shape):
            if i == axis:
                if _shape[i] is None or val is None:
                    _shape[i] = None
                else:
                    _shape[i] += val
            else:
                if _shape[i] is None or val is None:
                    _shape[i] = _shape[i] or val
                else:
                    if not _shape[i] == val:
                        raise ValueError('Inconsistent shape')
    return _shape


class Concat(object):
    """Implement Concat layer in Theano

    See :any: `BaseConcat` for detail.
    """
    def _build(self, var_list):
        if len(var_list) < 2:
            raise ValueError('var_list must contain more than 1 tensor')
        axis = self.args['axis']

        tensor_list = [var.unwrap() for var in var_list]
        shape_list = [var.shape for var in var_list]
        shape = _compute_concat_shape(shape_list, axis)
        output = T.concatenate(tensor_list=tensor_list, axis=axis)
        return Tensor(output, shape=shape, name='output')
