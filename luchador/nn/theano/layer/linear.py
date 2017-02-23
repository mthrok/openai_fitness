"""Implement Layer classes in Theano"""
from __future__ import division
from __future__ import absolute_import

import logging

import theano.tensor as T

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
    """Implement Dense layer in Theano.

    See :any:`BaseDense` for detail.
    """
    def _instantiate_parameters(self, n_inputs, dtype):
        initializers = _get_initializers(self.args.get('initializers') or {})

        w_shape = (n_inputs, self.args['n_nodes'])
        w_init = initializers['weight']
        self._add_parameter('weight', scope.get_variable(
            name='weight', shape=w_shape, initializer=w_init, dtype=dtype))

        if self.args['with_bias']:
            b_shape = (self.args['n_nodes'],)
            b_init = initializers['bias']
            self._add_parameter('bias', scope.get_variable(
                name='bias', shape=b_shape, initializer=b_init, dtype=dtype))

    def _build(self, input_tensor):
        input_shape = input_tensor.shape

        if not len(input_shape) == 2:
            raise ValueError('Input tensor must be 2D. '
                             'Insted of {}'.format(len(input_shape)))

        if not self._parameter_variables:
            self._instantiate_parameters(input_shape[1], input_tensor.dtype)

        weight = self._get_parameter('weight').unwrap()
        output = T.dot(input_tensor.unwrap(), weight)

        if self.args['with_bias']:
            bias = self._get_parameter('bias').unwrap()
            output = output + bias
        output_shape = (input_shape[0], self.args['n_nodes'])
        return wrapper.Tensor(output, shape=output_shape, name='output')
