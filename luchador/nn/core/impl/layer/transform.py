"""Define common interface for Layer classes"""
from __future__ import division
from __future__ import absolute_import

import logging

from luchador.configure import get_nn_conv_format
from ...base import BaseLayer
from ...backend import layer
from ...backend import ops
__all__ = ['Flatten', 'Concat', 'Reshape']
# pylint: disable=abstract-method

_LG = logging.getLogger(__name__)


class Flatten(layer.Flatten, BaseLayer):
    """Reshape 4D tensor into 2D tensor"""
    def __init__(self, scope='Flatten'):
        super(Flatten, self).__init__(scope=scope)


class Concat(layer.Concat, BaseLayer):
    """Concatenate multiple Tensors"""
    def __init__(self, axis=1, scope='Concat'):
        super(Concat, self).__init__(axis=axis, scope=scope)


def _convert_to_running_format(shape, given_format):
    running_format = get_nn_conv_format()
    if given_format == running_format:
        return shape
    if running_format == 'NCHW':
        _LG.info('  * Converting `shape` to NCHW')
        return (shape[0], shape[3], shape[1], shape[2])
    _LG.info('  * Converting `shape` to NHWC')
    return (shape[0], shape[2], shape[3], shape[1])


class Reshape(BaseLayer):
    """Reshape variable

    Parameters
    ----------
    shape : tuple
        New shape

    shape_format : None or str
        If you are providing ``shape`` constructor argument in YAML file, and
        reshaping variable to feed it ro convolution layer, you cannot tell in
        which convolution format luchador will use at runtime. So by adding
        ``shape_format`` which describes which convolution format it adopts,
        the layer shuffle the ``shape`` automatically.

    scope : str
        Scope of layer.
    """
    def __init__(self, shape, shape_format=None, scope='Reshape'):
        super(Reshape, self).__init__(
            shape=shape, scope=scope, shape_format=shape_format)

    def _build(self, input_tensor):
        given_format = self.args['shape_format']
        shape, scope = self.args['shape'], self.args['scope']
        if given_format:
            shape = _convert_to_running_format(shape, given_format)
        return ops.reshape(input_tensor, shape, name=scope)
