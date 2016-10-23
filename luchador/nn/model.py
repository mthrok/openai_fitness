from __future__ import absolute_import

from luchador.common import get_subclasses
from .core import scope as scp

__all__ = [
    'BaseModel', 'get_model', 'Sequential',
]


class BaseModel(object):
    """Base Model class"""
    pass


def get_model(name):
    """Get Model class by name"""
    for Class in get_subclasses(BaseModel):
        if Class.__name__ == name:
            return Class
    raise ValueError('Unknown model: {}'.format(name))


###############################################################################
class LayerConfig(object):
    """Class to hold complementary info for Layer class"""
    def __init__(self, layer, scope, input=None, output=None):
        self.layer = layer
        self.scope = scope
        self.input = input
        self.output = output

    def __eq__(self, other):
        if isinstance(other, LayerConfig):
            return self.layer == other.layer and self.scope == other.scope
        return NotImplemented

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return repr({
            'scope': self.scope,
            'layer': self.layer,
            'input': self.input,
            'output': self.output,
        })


class Sequential(BaseModel):
    def __init__(self):
        super(Sequential, self).__init__()
        # Layer configurations
        self.layer_configs = []
        # I/O tensors of the model
        self.input = None
        self.output = None

    ###########################################################################
    def add_layer(self, layer, scope=None):
        """Add layer to model

        Args:
          layer (Layer): Layer to add
          scope (str): Variable scope, used when instantiating layer.
        """
        self.layer_configs.append(LayerConfig(
            layer=layer,
            scope=scope,
            input=None,
            output=None,
        ))
        return self

    ###########################################################################
    # Functions for model-wise copy and concatenation
    def __iadd__(self, other):
        """Append layers from another model after this model
        Args:
          other (Sequential): Sequential to be concatenated

        Returns:
          Sequential: Updated model
        """
        for cfg in other.layer_configs:
            self.add_layer(layer=cfg.layer, scope=cfg.scope)
        return self

    def __add__(self, other):
        """Create new model which contains layers from other model after this model
        Args:
          other (Sequential): Sequential to be concatenated

        Returns:
          Sequential: Resulting new model
        """
        new_model = type(self)()
        new_model += self
        new_model += other
        return new_model

    ###########################################################################
    # Model equality
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.layer_configs == other.layer_configs
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    ###########################################################################
    # Functions for building actual computation graphs
    def __call__(self, input_tensor):
        return self.build(input_tensor)

    def build(self, input_tensor):
        """Build the model on top of input tensor"""
        tensor = self.input = input_tensor
        for cfg in self.layer_configs:
            cfg.input = tensor
            with scp.variable_scope(cfg.scope or scp.get_variable_scope()):
                tensor = cfg.layer(tensor)
            cfg.output = tensor
        self.output = tensor
        return self.output

    ###########################################################################
    # Functions for retrieving variables, tensors and operations
    def get_parameter_variables(self):
        """Get Variable objects consisting the parameters of this model"""
        ret = []
        for cfg in self.layer_configs:
            ret.extend(cfg.layer.parameter_variables.values())
        return ret

    def get_output_tensors(self):
        """Get Tensor objects which represent the output of each layer"""
        return [cfg.output for cfg in self.layer_configs]

    def get_update_operations(self):
        """Get Update opretaions from each layer"""
        return [cfg.layer.get_update_operation() for cfg in self.layer_configs]

    ###########################################################################
    def serialize(self):
        """Serialize model configuration"""
        return {
            'model_type': self.__class__.__name__,
            'layer_configs': [{
                'scope': cfg.scope,
                'layer': cfg.layer.serialize(),
            } for cfg in self.layer_configs]
        }

    ###########################################################################
    def __repr__(self):
        return repr({
            'model_type': self.__class__.__name__,
            'layer_configs': self.layer_configs,
        })
