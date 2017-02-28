"""Implement LayerMixin fot Theano layers"""
from __future__ import absolute_import

import abc
import logging

from .. import scope

_LG = logging.getLogger(__name__)


class LayerMixin(object):
    """Mixin for scoped build"""
    def build(self, input_tensor):
        """Build layer computation graph on top of the given tensor

        Parameters
        ----------
        input_tensor : Tensor
            Tensor object. Requirement for this object (such as shape and
            dtype) varies from Layer types.

        Returns
        -------
        Tensor
            Tensor which holds the output of built computation
        """
        _LG.info(
            '  Building layer %s on %s', type(self).__name__, input_tensor)

        self.input = input_tensor
        with scope.variable_scope(self.args['name']):
            self.output = self._build(input_tensor)
            return self.output

    @abc.abstractmethod
    def _build(self, input_tensor):
        """Actual build method"""
        raise NotImplementedError(
            '`_build` method is not implemented for {}'.format(self.__class__)
        )
