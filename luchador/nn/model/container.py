"""Define Container class which can build model stiching Sequeitla model"""
from __future__ import absolute_import

from collections import OrderedDict

from .base_model import BaseModel


class Container(BaseModel):
    """Data structure for handling multiple network architectures at once

    Using this class and build utility functions make it easy to build
    multi-branching-merging network.
    """
    def __init__(self):
        super(Container, self).__init__()
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

    def get_parameter_variables(self):
        """Get parameter Variables

        Returns
        -------
        list
            List of Variables from interanal models.
        """
        ret = []
        for name_ in self.models.keys():
            ret.extend(self.models[name_].get_parameter_variables())
        return ret

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
    def __repr__(self):
        return repr({self.__class__.__name__: self.models})
