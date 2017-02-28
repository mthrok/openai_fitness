"""Implement Layer classes in Tensorflow"""
from __future__ import division
from __future__ import absolute_import

import logging

import tensorflow as tf

from ...base import layer as base_layer
from ..wrapper import Tensor
from .common import LayerMixin

__all__ = [
    'NHWC2NCHW', 'NCHW2NHWC',
]

_LG = logging.getLogger(__name__)


class NHWC2NCHW(LayerMixin, base_layer.BaseNHWC2NCHW):
    """See :any:`BaseNHWC2NCHW` for detail."""
    def _build(self, input_tensor):
        output = tf.transpose(input_tensor.unwrap(), perm=(0, 3, 1, 2))
        return Tensor(output, name='output')


class NCHW2NHWC(LayerMixin, base_layer.BaseNCHW2NHWC):
    """See :any:`BaseNCHW2NHWC` for detail."""
    def _build(self, input_tensor):
        output = tf.transpose(input_tensor.unwrap(), perm=(0, 2, 3, 1))
        return Tensor(output, name='output')
