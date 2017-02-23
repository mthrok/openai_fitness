"""Implement Layer classes in Tensorflow"""
from __future__ import division
from __future__ import absolute_import

import logging

import tensorflow as tf

from ...base import layer as base_layer
from ...base import getter
from .. import scope, wrapper

__all__ = ['Dense']

_LG = logging.getLogger(__name__)


def _get_initializers(config):
    """Get initializers for Conv2D

    Parameters
    ----------
    config : dict
        weight : dict
            Initializer configuration for ``weight`` parameter. If not present,
            :func:`luchador.nn.theano.initializer.Xavier` is used.
        bias : dict
            Initializer configuration for ``bias`` parameter. If not present,
            :func:`luchador.nn.theano.initializer.Constant` with
            ``value = 0.1`` is used.

    Returns
    -------
    dict
        Resulting initializers for ``weight`` and ``bias``
    """
    ret = {}

    cfg = config.get('weight', {'typename': 'Xavier'})
    type_ = cfg['typename']
    ret['weight'] = getter.get_initializer(type_)(**cfg.get('args', {}))

    cfg = config.get('bias', {'typename': 'Constant', 'args': {'value': 0.1}})
    type_ = cfg['typename']
    ret['bias'] = getter.get_initializer(type_)(**cfg.get('args', {}))
    return ret


class Dense(base_layer.BaseDense):
    """Implement Dense layer in Tensorflow.

    See :any:`BaseDense` for detail.
    """
    def _instantiate_parameters(self, n_inputs, dtype):
        initializers = _get_initializers(self.args.get('initializers') or {})

        w_shape = (n_inputs, self.args['n_nodes'])
        weight = scope.get_variable(
            name='weight', shape=w_shape, dtype=dtype,
            initializer=initializers['weight'])
        self._add_parameter('weight', weight)

        if self.args['with_bias']:
            b_shape = (self.args['n_nodes'],)
            bias = scope.get_variable(
                name='bias', shape=b_shape, dtype=dtype,
                initializer=initializers['bias'])
            self._add_parameter('bias', bias)

    def _build(self, input_tensor):
        if not self._parameter_variables:
            self._instantiate_parameters(
                input_tensor.shape[1], input_tensor.dtype)

        weight = self._get_parameter('weight').unwrap()
        output = tf.matmul(input_tensor.unwrap(), weight)

        if self.args['with_bias']:
            bias = self._get_parameter('bias').unwrap()
            output = tf.add(output, bias, name='output')
        return wrapper.Tensor(output, name='output')
