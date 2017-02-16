"""Define Container class which can build model stiching Sequeitla model"""
from __future__ import absolute_import

from collections import OrderedDict

from .base_model import BaseModel


class Container(BaseModel):
    """Data structure for handling multiple network architectures at once

    As a result, using this class and build utility functions,
    multi-branching-merging network is built easily.
    """
    def __init__(self):
        super(Container, self).__init__()
        self.input = None
        self.output = None
        self.models = OrderedDict()

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

    def get_parameter_variables(self, name=None):
        """Get Variable objects consisting the parameters of this model

        Parameters
        ----------
        name : str or None
            Name of model from which parameter variables are retrieved.
            If None, parameters from all the models are fetched.

        Returns
        -------
        list
            List of Variables from layer parameters
        """
        if name:
            return self.models[name].get_parameter_variables()

        ret = []
        for name_ in self.models.keys():
            ret.extend(self.models[name_].get_parameter_variables())
        return ret

    def get_output_tensors(self, name=None):
        """Get Tensor s which represent the output of each layer of this model

        Parameters
        ----------
        name : str or None
            Name of model from which parameter variables are retrieved.
            If None, parameters from all the models are fetched.

        Returns
        -------
        list
            List of Tensors each of which hold output from layer
        """
        if name:
            return self.models[name].get_output_tensors()

        ret = []
        for name_ in self.models.keys():
            ret.extend(self.models[name_].get_output_tensors())
        return ret

    def get_update_operations(self, name=None):
        """Get update opretaions from each layer of this model

        Parameters
        ----------
        name : str or None
            Name of model from which parameter variables are retrieved.
            If None, parameters from all the models are fetched.

        Returns
        -------
        list
            List of update operations from each layer
        """
        if name:
            return self.models[name].get_update_operations()

        ret = []
        for name_ in self.models.keys():
            ret.extend(self.models[name_].get_update_operations())
        return ret

    ###########################################################################
    def __repr__(self):
        return repr({
            'model_type': self.__class__.__name__,
            'models': self.models,
        })
