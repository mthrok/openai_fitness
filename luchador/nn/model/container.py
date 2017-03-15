"""Define Container class which can build model stiching Sequeitla model"""
from __future__ import absolute_import

from collections import OrderedDict

from .base_model import BaseModel


class Container(BaseModel):
    """Data structure for handling multiple network architectures at once

    Using this class and build utility functions make it easy to build
    multi-branching-merging network.
    """
    def __init__(self, name=None):
        super(Container, self).__init__(name=name)
        self.models = OrderedDict()
        self.input = None
        self.output = None

    def add_model(self, name, model):
        """Add model.

        Parameters
        ----------
        name : str
            Name of model to store.

        model : Model
            Model object.
        """
        self.models[name] = model
        return self

    def get_parameters_to_train(self):
        """Get parameter Variables to be fet to gradient computation.

        Returns
        -------
        list
            List of Variables from interanal models.
        """
        ret = []
        for name_ in self.models.keys():
            ret.extend(self.models[name_].get_parameters_to_train())
        return ret

    def get_parameters_to_serialize(self):
        """Get parameter Variables to be serialized.

        Returns
        -------
        list
            List of Variables from internal models.
        """
        ret = []
        for name_ in self.models.keys():
            ret.extend(self.models[name_].get_parameters_to_serialize())
        return ret

    def get_output_tensors(self):
        """Get Tensor s which represent the output of each layer of this model

        Returns
        -------
        list
            List of Tensors each of which hold output from layer
        """
        ret = []
        for name_ in self.models.keys():
            ret.extend(self.models[name_].get_output_tensors())
        return ret

    def get_update_operations(self):
        """Get update opretaions from each layer of this model

        Returns
        -------
        list
            List of update operations from each layer
        """
        ret = []
        for name_ in self.models.keys():
            ret.extend(self.models[name_].get_update_operations())
        return ret

    ###########################################################################
    def __getitem__(self, key):
        """Get an underlying model with name"""
        return self.models[key]

    def __setitem__(self, key, model):
        """Set model"""
        self.add_model(key, model)

    def __repr__(self):
        return repr({self.__class__.__name__: self.models})
