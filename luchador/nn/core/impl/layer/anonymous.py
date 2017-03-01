"""Define AnonymousLayer classes"""
from __future__ import division
from __future__ import absolute_import

from ...base import BaseLayer
from ...backend import Tensor

__all__ = ['Anonymous']
# pylint: disable=abstract-method


class Anonymous(BaseLayer):
    def __init__(self, exp='', name='Anonymous'):
        super(Anonymous, self).__init__(name=name, exp=exp)

    def _build(self, x):
        y = eval(self.args['exp'])
        return Tensor(tensor=y.unwrap(), shape=y.shape, name='output')
