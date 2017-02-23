"""Common functionalities for Layer module

Should not be exported.
"""
from __future__ import division
from __future__ import absolute_import

from ...base import getter
from .. import initializer


def get_initializers(cfg, with_bias):
    """Initializer for Dense and Conv2D"""
    w_cfg = cfg.get('weight')
    ret = {}
    ret['weight'] = (
        getter.get_initializer(w_cfg['typename'])(**w_cfg['args'])
        if w_cfg else initializer.Xavier()
    )

    if with_bias:
        _cfg = cfg.get('bias')
        ret['bias'] = (
            getter.get_initializer(_cfg['typename'])(**_cfg['args'])
            if _cfg else initializer.Constant(0.1)
        )
    return ret
