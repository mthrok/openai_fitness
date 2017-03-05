"""Define AnonymousLayer classes"""
from __future__ import division
from __future__ import absolute_import

from ...base import BaseLayer
from ...backend import wrapper, ops
from .. import random

__all__ = ['Anonymous']
# pylint: disable=abstract-method


def _parse_input_tensors(input_tensor, *args, **kwargs):
    if input_tensor:
        if kwargs:
            raise ValueError(
                'Anonymouse layer accepsts either positional parameters '
                'or keyword arguments, not both.')
        if args:
            return [input_tensor] + args
        return input_tensor
    return kwargs


def _get_safe_function(input_tensor, *args, **kwargs):
    if args and kwargs:
        raise ValueError(
            'Anonymouse layer accepsts either positional parameters '
            'or keyword arguments, not both.')
    maps = {
        'x': _parse_input_tensors(input_tensor, *args, **kwargs),
        'True': True,
        'False': False,
        'NormalRandom': random.NormalRandom,
        'UniformRandom': random.UniformRandom,
    }
    for key in ops.__all__:
        maps[key] = getattr(ops, key)
    return maps


class Anonymous(BaseLayer):
    """Run externally-provided computation on input tensor"""
    def __init__(self, exp, name='Anonymous'):
        super(Anonymous, self).__init__(name=name, exp=exp)

    def __call__(self, input_tensor, *args, **kwargs):
        """Convenience method to call `build`"""
        return self.build(input_tensor, *args, **kwargs)

    def build(self, input_tensor, *args, **kwargs):
        """Build Anonymous layer


        """
        return self._build(input_tensor, *args, **kwargs)

    def _build(self, input_tensor, *args, **kwargs):
        # pylint: disable=eval-used
        local = _get_safe_function(input_tensor, *args, **kwargs)
        y = eval(self.args['exp'], {'__builtins__': None}, local)
        return wrapper.Tensor(tensor=y.unwrap(), shape=y.shape, name='output')
