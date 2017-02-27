"""Utility functions commonly used in submodules"""
from __future__ import absolute_import

__all__ = ['nchw2nhwc', 'nhwc2nchw']


def nchw2nhwc(shape):
    """Convert shape format from NCHW to NHWC"""
    return (shape[0], shape[2], shape[3], shape[1])


def nhwc2nchw(shape):
    """Convert shape format from NHWC to NCHW"""
    return (shape[0], shape[3], shape[1], shape[2])
