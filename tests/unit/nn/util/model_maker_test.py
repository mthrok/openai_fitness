"""Test nn.util module"""
from __future__ import absolute_import

import numpy as np

from luchador import nn
from tests.unit import fixture

# pylint: disable=invalid-name


class MakeIOTest(fixture.TestCase):
    """Test make_io_node function"""
    def test_single_node(self):
        """Can make/fetch single node"""
        name = 'input_state'
        input_config = {
            'typename': 'Input',
            'args': {
                'shape': (32, 5),
                'name': name,
            },
        }
        input_config_reuse = {
            'typename': 'Input',
            'reuse': True,
            'name': name,
        }
        with nn.variable_scope(self.get_scope()):
            input1 = nn.make_io_node(input_config)
            input2 = nn.make_io_node(input_config_reuse)
            input_ = nn.get_input(name=name)
            self.assertIs(input1, input2)
            self.assertIs(input1, input_)

    def test_list_nodes(self):
        """Can make/fetch list of nodes"""
        names = ['input_state_1', 'input_state_2']
        config0 = {
            'typename': 'Input',
            'args': {
                'shape': (32, 5),
                'name': names[0],
            },
        }
        config1 = {
            'typename': 'Input',
            'args': {
                'shape': (32, 5),
                'name': names[1],
            },
        }
        with nn.variable_scope(self.get_scope()):
            inputs = nn.make_io_node([config0, config1])
            input0 = nn.get_input(name=names[0])
            input1 = nn.get_input(name=names[1])
            self.assertIs(input0, inputs[0])
            self.assertIs(input1, inputs[1])

    def test_map_nodes(self):
        """Can make/fetch dict of nodes"""
        names = ['input_state_1', 'input_state_2']
        config0 = {
            'typename': 'Input',
            'args': {
                'shape': (32, 5),
                'name': names[0],
            },
        }
        config1 = {
            'typename': 'Input',
            'args': {
                'shape': (32, 5),
                'name': names[1],
            },
        }
        with nn.variable_scope(self.get_scope()):
            inputs = nn.make_io_node({'config0': config0, 'config1': config1})
            input0 = nn.get_input(name=names[0])
            input1 = nn.get_input(name=names[1])
            self.assertIs(input0, inputs['config0'])
            self.assertIs(input1, inputs['config1'])


class ModelMakerTest(fixture.TestCase):
    """Test make_model functions"""
    def test_make_layer_with_reuse(self):
        """make_layer sets parameter variables correctly"""
        shape, scope, name = (3, 4), self.get_scope(), 'Dense'
        layer_config = {
            'typename': 'Dense',
            'args': {
                'n_nodes': 5,
                'name': 'Dense',
            },
            'parameters': {
                'weight': {
                    'typename': 'Variable',
                    'name': '{}/{}/weight'.format(scope, name)
                },
                'bias': {
                    'typename': 'Variable',
                    'name': '{}/{}/bias'.format(scope, name)
                },
            }
        }

        with nn.variable_scope(scope):
            layer1 = nn.layer.Dense(n_nodes=5, name=name)
            tensor = nn.Input(shape=shape)
            out1 = layer1(tensor)

        layer2 = nn.make_node(layer_config)
        out2 = layer2(tensor)

        for key in ['weight', 'bias']:
            var1 = layer1.get_parameter_variable(key)
            var2 = layer2.get_parameter_variable(key)
            self.assertIs(var1, var2)

        session = nn.Session()
        session.initialize()

        input_val = np.random.rand(*shape)
        out1, out2 = session.run(
            outputs=[out1, out2],
            inputs={tensor: input_val}
        )

        np.testing.assert_almost_equal(
            out1, out2
        )

    def test_make_layer_with_reuse_in_scope(self):
        """make_layer sets parameter variables correctly"""
        shape, scope1, scope2, name = (3, 4), self.get_scope(), 'foo', 'dense'
        layer_config = {
            'typename': 'Dense',
            'args': {
                'n_nodes': 5,
                'name': name,
            },
            'parameters': {
                'weight': {
                    'typename': 'Variable',
                    'name': '{}/{}/weight'.format(scope2, name),
                },
                'bias': {
                    'typename': 'Variable',
                    'name': '{}/{}/bias'.format(scope2, name),
                },
            }
        }
        with nn.variable_scope(scope1):
            with nn.variable_scope(scope2):
                layer1 = nn.layer.Dense(n_nodes=5, name=name)
                tensor = nn.Input(shape=shape)
                out1 = layer1(tensor)

            layer2 = nn.make_node(layer_config)
            out2 = layer2(tensor)

        for key in ['weight', 'bias']:
            var1 = layer1.get_parameter_variable(key)
            var2 = layer2.get_parameter_variable(key)
            self.assertIs(var1, var2)

        session = nn.Session()
        session.initialize()

        input_val = np.random.rand(*shape)
        out1, out2 = session.run(
            outputs=[out1, out2],
            inputs={tensor: input_val}
        )

        np.testing.assert_almost_equal(
            out1, out2
        )
