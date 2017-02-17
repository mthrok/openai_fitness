"""Define base network model structure and fetch method"""
from __future__ import absolute_import

import abc


class BaseModel(object):  # pylint: disable=too-few-public-methods
    """Base Model class"""
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(BaseModel, self).__init__()

        self.input = None
        self.output = None
