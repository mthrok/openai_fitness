from __future__ import absolute_import

import luchador.nn as nn
from tests.unit import fixture


class LayerInterfaceTest(fixture.TestCase):
    """Test layer interface"""
    def test_non_parametric_layers(self):
        """Compnents consisting layer are retrieved"""
        layer_names = [
            'ReLU', 'Sigmoid', 'Tanh', 'Sin', 'Cos', 'Softmax', 'Softplus',
        ]
        for name in layer_names:
            self._test_layer_io(name, input_shape=(32, 10))

        layer_names = [
            'Flatten', 'NHWC2NCHW', 'NCHW2NHWC',
        ]
        for name in layer_names:
            self._test_layer_io(name, input_shape=(32, 4, 8, 8))

    def _test_layer_io(self, layer_name, input_shape):
        scope = '{}/{}'.format(self.get_scope(), layer_name)
        with nn.variable_scope(scope) as vs:
            input_ = nn.Input(shape=input_shape, name='input')
            layer = nn.get_layer(layer_name)()
            output = layer(input_)

        with nn.variable_scope(vs, reuse=True):
            self.assertIs(output, nn.get_tensor('output'))
            self.assertIs(input_, nn.get_input('input'))

    def test_dense(self):
        """Compnents consisting Dense layer are retrieved"""
        scope = self.get_scope()
        with nn.variable_scope(scope) as vs:
            input_ = nn.Input(shape=(32, 5), name='input')
            layer = nn.get_layer('Dense')(n_nodes=4, with_bias=True)
            output = layer(input_)
            weight = layer.get_parameter_variables('weight')
            bias = layer.get_parameter_variables('bias')

        with nn.variable_scope(vs, reuse=True):
            self.assertIs(weight, nn.get_variable('weight'))
            self.assertIs(bias, nn.get_variable('bias'))
            self.assertIs(output, nn.get_tensor('output'))
            self.assertIs(input_, nn.get_input('input'))

    def test_conv2d(self):
        """Compnents consisting Conv2D layer are retrieved"""
        scope = self.get_scope()
        with nn.variable_scope(scope) as vs:
            input_ = nn.Input(shape=(32, 4, 8, 8), name='input')
            layer = nn.get_layer('Conv2D')(
                filter_height=4, filter_width=4, n_filters=4,
                strides=1, with_bias=True)
            output = layer(input_)
            weight = layer.get_parameter_variables('weight')
            bias = layer.get_parameter_variables('bias')

        with nn.variable_scope(vs, reuse=True):
            self.assertIs(weight, nn.get_variable('weight'))
            self.assertIs(bias, nn.get_variable('bias'))
            self.assertIs(output, nn.get_tensor('output'))
            self.assertIs(input_, nn.get_input('input'))

    def test_true_div(self):
        """Compnents consisting truediv layer are retrieved"""
        scope = self.get_scope()
        with nn.variable_scope(scope):
            input_ = nn.Input(shape=(32, 4, 8, 8), name='input')
            layer = nn.get_layer('TrueDiv')(denom=1.0)
            output = layer(input_)

            self.assertIs(output, nn.get_tensor('output'))
            self.assertIs(input_, nn.get_input('input'))

    def test_mean(self):
        """Compnents consisting Mean layer are retrieved"""
        scope = self.get_scope()
        with nn.variable_scope(scope):
            input_ = nn.Input(shape=(32, 4, 8, 8), name='input')
            layer = nn.get_layer('Mean')(axis=[1, 2])
            output = layer(input_)

            self.assertIs(output, nn.get_tensor('output'))
            self.assertIs(input_, nn.get_input('input'))

    def test_tile(self):
        """Compnents consisting Tile layer are retrieved"""
        scope = self.get_scope()
        with nn.variable_scope(scope):
            input_ = nn.Input(shape=(32,), name='input')
            layer = nn.get_layer('Tile')(pattern=(1, 2))
            output = layer(input_)

            self.assertIs(output, nn.get_tensor('output'))
            self.assertIs(input_, nn.get_input('input'))

    def test_concat(self):
        """Compnents consisting Concat layer are retrieved"""
        scope = self.get_scope()
        with nn.variable_scope(scope):
            input_ = [
                nn.Input(shape=(32, 4), name='input'),
                nn.Input(shape=(32, 5), name='input'),
            ]
            layer = nn.get_layer('Concat')(axis=1)
            output = layer(input_)
            self.assertIs(output, nn.get_tensor('output'))

    def test_add(self):
        """Compnents consisting Add layer are retrieved"""
        scope = self.get_scope()
        with nn.variable_scope(scope):
            input_ = [
                nn.Input(shape=(32, 4), name='input'),
                nn.Input(shape=(32, 4), name='input'),
            ]
            layer = nn.get_layer('Add')()
            output = layer(input_)
            self.assertIs(output, nn.get_tensor('output'))

    def test_sub(self):
        """Compnents consisting Sub layer are retrieved"""
        scope = self.get_scope()
        with nn.variable_scope(scope):
            input_ = [
                nn.Input(shape=(32, 4), name='input'),
                nn.Input(shape=(32, 4), name='input'),
            ]
            layer = nn.get_layer('Sub')()
            output = layer(input_)
            self.assertIs(output, nn.get_tensor('output'))

    def test_bn(self):
        """Compnents consisting BatchNormalization layer are retrieved"""
        scope = self.get_scope()
        with nn.variable_scope(scope) as vs:
            input_ = nn.Input(shape=(32, 4), name='input')
            layer = nn.get_layer('BatchNormalization')()
            output = layer(input_)
            mean = layer.get_parameter_variables('mean')
            var = layer.get_parameter_variables('var')
            scale = layer.get_parameter_variables('scale')
            offset = layer.get_parameter_variables('offset')
            update = layer.get_update_operation()

        with nn.variable_scope(vs, reuse=True):
            self.assertIs(mean, nn.get_variable('mean'))
            self.assertIs(var, nn.get_variable('var'))
            self.assertIs(scale, nn.get_variable('scale'))
            self.assertIs(offset, nn.get_variable('offset'))
            self.assertIs(output, nn.get_tensor('output'))
            self.assertIs(update, nn.get_operation('bn_update'))
