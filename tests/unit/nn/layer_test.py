from __future__ import absolute_import

import unittest

from luchador.nn import (
    get_layer,
    Dense,
    Conv2D,
    ReLU,
    Flatten,
    TrueDiv,
    BatchNormalization,
)

PARAMETERIZED_LAYER_CLASSES = (
    Dense,
    Conv2D,
    TrueDiv,
    BatchNormalization,
)

FIXED_LAYER_CLASSES = (
    Flatten,
    ReLU,
)


LAYERS = PARAMETERIZED_LAYER_CLASSES + FIXED_LAYER_CLASSES
N_LAYERS = 6

ARGS1 = {
    'Dense': {
        'n_nodes': 100,
    },
    'Conv2D': {
        'filter_height': 8,
        'filter_width': 8,
        'n_filters': 32,
        'strides': 1,
        'padding': 'VALID',
    },
    'TrueDiv': {
        'denom': 1,
    },
    'BatchNormalization': {
        'center': 0.0,
        'scale': 1.0,
    },
}

ARGS2 = {
    'Dense': {
        'n_nodes': 200,
    },
    'Conv2D': {
        'filter_height': 4,
        'filter_width': 4,
        'n_filters': 16,
        'strides': 2,
        'padding': 'SAME',
    },
    'TrueDiv': {
        'denom': 255
    },
    'BatchNormalization': {
        'center': 0.5,
        'scale': 1.0,
    },
}


def make_parameterized_layers(args):
    return [Layer(**args[Layer.__name__])
            for Layer in PARAMETERIZED_LAYER_CLASSES]


def make_fixed_layers():
    return [Layer() for Layer in FIXED_LAYER_CLASSES]


class LayerTest(unittest.TestCase):
    longMessage = True

    def test_layer_test_coverage(self):
        """All initializers are tested"""
        self.assertEqual(
            len(LAYERS), N_LAYERS,
            'Number of layers are changed. (New layer is added?) '
            'Fix unittest to cover new layers'
        )

    def test_parameterized_layer_equality(self):
        """Parameterized layers with same arguments must be equal"""
        layers1 = make_parameterized_layers(ARGS1)
        layers2 = make_parameterized_layers(ARGS1)
        for l1, l2 in zip(layers1, layers2):
            self.assertEqual(
                l1, l2,
                '\n{} layers initialized with the same arguments '
                'must be equal.'.format(type(l1)))

    def test_parameterized_layer_inequality(self):
        """Parameterized layers with different arguments must not be equal"""
        layers1 = make_parameterized_layers(ARGS1)
        layers2 = make_parameterized_layers(ARGS2)
        for l1, l2 in zip(layers1, layers2):
            self.assertNotEqual(
                l1, l2,
                '\n{} layers initialized with the differnet arguments '
                'must not be equal.'.format(type(l1)))

    def test_fixed_layer_equality(self):
        """Instances of not parmaeterized layers must be equal"""
        layers1 = make_fixed_layers()
        layers2 = make_fixed_layers()
        for l1, l2 in zip(layers1, layers2):
            self.assertEqual(
                l1, l2, '\n{} layers must be eqauel'.format(type(l1)))

    def test_layer_serialize(self):
        """`serialize` method should return constructor arguments """
        layers = make_parameterized_layers(ARGS1)
        for layer in layers:
            args = layer.serialize()
            layer_name = args['name']
            given_args = ARGS1[layer_name]
            for key, expected in given_args.items():
                found = args['args'][key]
                self.assertEqual(
                    expected, found,
                    '\nArgument {} serialized from {} are different from '
                    'what was given in constructor.'
                    .format(key, layer_name))

    def test_layer_equality_serialize(self):
        """Layers recreated with serialized arguments equal to originals"""
        for layer0 in make_parameterized_layers(ARGS1):
            args = layer0.serialize()
            layer1 = get_layer(args['name'])(**args['args'])
            expected = layer0
            found = layer1
            self.assertEqual(
                expected, found,
                '\nLayer recreated from serialize arguments must '
                'equal to the original.')
