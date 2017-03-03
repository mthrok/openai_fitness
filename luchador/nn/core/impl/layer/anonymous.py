"""Define AnonymousLayer classes"""
from __future__ import division
from __future__ import absolute_import

from ...base import BaseLayer
from ...backend import wrapper, ops

__all__ = ['Anonymous']
# pylint: disable=abstract-method


def _get_safe_function(input_tensor):
    maps = {
        'x': input_tensor,
        'True': True,
        'False': False,
    }
    for key in ops.__all__:
        maps[key] = getattr(ops, key)
    return maps


class Anonymous(BaseLayer):
    """Run externally-provided computation on input tensor"""
    def __init__(self, exp, name='Anonymous'):
        super(Anonymous, self).__init__(name=name, exp=exp)

    def _build(self, input_tensor):
        # pylint: disable=eval-used
        local = _get_safe_function(input_tensor)
        y = eval(self.args['exp'], {'__builtins__': None}, local)
        return wrapper.Tensor(tensor=y.unwrap(), shape=y.shape, name='output')
