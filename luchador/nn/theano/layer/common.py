"""Functions common to layer sub modules.

This module is not meant to be exported.
"""
from __future__ import division
from __future__ import absolute_import


from ...base import getter
from .. import initializer


def get_initializers(cfg, with_bias):
    """Fetch initializers for Dense and Conv2D

    Parameters
    ----------
    cfg : dict
        Configuration for weight and bias initializer

        weight
            Configuration for weight initializer
        bias
            Configuration for bias initilaizer

    with_bias : bool
        If True bias initializer is also included in return

    Returns
    -------
    dict
        Dictionary containing the resulting initializers

    """
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
