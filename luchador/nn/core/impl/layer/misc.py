"""Define common interface for Layer classes"""
from __future__ import division
from __future__ import absolute_import

from ...base import BaseLayer
from ...backend import layer

__all__ = ['NHWC2NCHW', 'NCHW2NHWC']
# pylint: disable=abstract-method


class NHWC2NCHW(layer.NHWC2NCHW, BaseLayer):
    """Convert NCHW data from to NHWC

    Rearrange the order of axes in 4D tensor from NHWC to NCHW

    Parameters
    ----------
    name : str
        Used as base scope when building parameters and output
    """
    def __init__(self, name='NHWC2NCHW'):
        super(NHWC2NCHW, self).__init__(name=name)


class NCHW2NHWC(layer.NCHW2NHWC, BaseLayer):
    """Convert NCHW data flow to NHWC

    Rearrange the order of axes in 4D tensor from NCHW to NHWC

    Parameters
    ----------
    name : str
        Used as base scope when building parameters and output
    """
    def __init__(self, name='NCHW2NHWC'):
        super(NCHW2NHWC, self).__init__(name=name)
