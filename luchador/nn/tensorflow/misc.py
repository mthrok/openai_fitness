"""Module for providing backend-common interface for misc task"""
from __future__ import absolute_import

from .wrapper import Operation

__all__ = ['build_sync_op']


def build_sync_op(source_vars, target_vars, name='sync'):
    """Build operation to copy values"""
    op = [
        target.unwrap().assign(source.unwrap())
        for source, target in zip(source_vars, target_vars)
    ]
    return Operation(op=op, name=name)
