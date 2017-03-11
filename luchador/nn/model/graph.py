"""Define multi-input-multi-output Model architecture"""
from __future__ import absolute_import

from .base_model import BaseModel

__all__ = ['Graph']


class Graph(BaseModel):
    """Network architecture for bundling multiple nodes"""
    def __init__(self, name=None):
        super(Graph, self).__init__(name=name)
        self.nodes = []

    def add_node(self, node):
        """Add node to graph

        Parameters
        ----------
        node : Node
            Node to add.
        """
        self.nodes.append(node)

    ###########################################################################
    # Functions for retrieving variables, tensors and operations
    def get_parameters_to_train(self):
        """Get parameter Variables to be fed to gradient computation.

        Returns
        -------
        list
            List of Variables from interanal models.
        """
        ret = []
        for node in self.nodes:
            ret.extend(node.get_parameters_to_train())
        return ret

    def get_parameters_to_serialize(self):
        """Get parameter Variables to be serialized.

        Returns
        -------
        list
            List of Variables from internal models.
        """
        ret = []
        for node in self.nodes:
            ret.extend(node.get_parameters_to_serialize())
        return ret

    def get_output_tensors(self):
        """Get Tensor s which represent the output of each layer of this model

        Returns
        -------
        list
            List of Tensors each of which hold output from layer
        """
        return [node.output for node in self.nodes]

    def get_update_operations(self):
        """Get update opretaions from each layer

        Returns
        -------
        list
            List of update operations from each layer
        """
        ret = []
        for node in self.nodes:
            ret.extend(node.get_update_operations())
        return ret

    ###########################################################################
    def __repr__(self):
        return repr({self.__class__.__name__: self.nodes})
