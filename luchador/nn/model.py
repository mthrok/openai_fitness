from __future__ import absolute_import

from .core import scope as scp

__all__ = ['Sequential']


class BaseModel(object):
    pass


class LayerConfig(object):
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
        return ("{{'scope': '{}', 'layer': {}, 'input': {}, 'output': {}}}"
                .format(self.scope, self.layer, self.input, self.output))


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
    def __call__(self, input):
        return self.build(input)

    def build(self, input):
        """Build the model on top of input tensor"""
        tensor = self.input = input
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
        return "{{'layer_configs': {}}}".format(self.layer_configs)
