from __future__ import absolute_import

import luchador.nn as nn
from tests.unit import fixture


class LayerInterfaceTest(fixture.TestCase):
    """Test layer interface"""
    def test_non_parametric_layers(self):
        """Compnents consisting layer are retrieved"""
        layer_names = [
            'ReLU', 'Sigmoid', 'Tanh', 'Softmax', 'Softplus',
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
            layer = nn.fetch_layer(layer_name)(scope=layer_name)
            output = layer(input_)

        with nn.variable_scope(vs, reuse=True):
            output_tensor_name = '{}/output'.format(layer_name)
            self.assertIs(input_, nn.get_input('input'))
            self.assertIs(output, nn.get_tensor(output_tensor_name))

    def test_dense(self):
        """Compnents consisting Dense layer are retrieved"""
        with nn.variable_scope(self.get_scope()) as vs:
            input_ = nn.Input(shape=(32, 5), name='input')
            layer = nn.fetch_layer('Dense')(
                n_nodes=4, with_bias=True, scope='Dense')
            output = layer(input_)
            weight = layer.get_parameter_variable('weight')
            bias = layer.get_parameter_variable('bias')

        with nn.variable_scope(vs, reuse=True):
            self.assertIs(weight, nn.get_variable('Dense/weight'))
            self.assertIs(bias, nn.get_variable('Dense/bias'))
            self.assertIs(output, nn.get_tensor('Dense/output'))
            self.assertIs(input_, nn.get_input('input'))

    def test_conv2d(self):
        """Compnents consisting Conv2D layer are retrieved"""
        scope = self.get_scope()
        with nn.variable_scope(scope) as vs:
            input_ = nn.Input(shape=(32, 4, 8, 8), name='input')
            layer = nn.fetch_layer('Conv2D')(
                filter_height=4, filter_width=4, n_filters=4,
                strides=1, with_bias=True, name='Conv2D')
            output = layer(input_)
            filters = layer.get_parameter_variable('filter')
            bias = layer.get_parameter_variable('bias')

        with nn.variable_scope(vs, reuse=True):
            self.assertIs(filters, nn.get_variable('Conv2D/filter'))
            self.assertIs(bias, nn.get_variable('Conv2D/bias'))
            self.assertIs(output, nn.get_tensor('Conv2D/output'))
            self.assertIs(input_, nn.get_input('input'))

    def test_conv2dtranspose(self):
        """Compnents consisting Conv2DTranspose layer are retrieved"""
        with nn.variable_scope(self.get_scope()) as vs:
            input_ = nn.Input(shape=(32, 4, 8, 8), name='input')
            layer = nn.fetch_layer('Conv2D')(
                filter_height=4, filter_width=4, n_filters=4,
                strides=1, with_bias=True, scope='Conv2D')
            output = layer(input_)
            layer = nn.fetch_layer('Conv2DTranspose')(
                filter_height=4, filter_width=4, n_filters=4,
                strides=1, with_bias=True, output_shape=input_.shape,
                scope='Conv2DT')
            output = layer(output)
            filters = layer.get_parameter_variable('filter')
            bias = layer.get_parameter_variable('bias')

        with nn.variable_scope(vs, reuse=True):
            self.assertIs(filters, nn.get_variable('Conv2DT/filter'))
            self.assertIs(bias, nn.get_variable('Conv2DT/bias'))
            self.assertIs(output, nn.get_tensor('Conv2DT/output'))
            self.assertIs(input_, nn.get_input('input'))

    def test_true_div(self):
        """Compnents consisting truediv layer are retrieved"""
        scope = self.get_scope()
        with nn.variable_scope(scope):
            input_ = nn.Input(shape=(32, 4, 8, 8), name='input')
            layer = nn.fetch_layer('TrueDiv')(denom=1.0, scope='TrueDiv')
            output = layer(input_)

            self.assertIs(output, nn.get_tensor('TrueDiv/output'))
            self.assertIs(input_, nn.get_input('input'))

    def test_concat(self):
        """Compnents consisting Concat layer are retrieved"""
        with nn.variable_scope(self.get_scope()):
            layer = nn.fetch_layer('Concat')(axis=1, scope='Concat')
            output = layer([
                nn.Input(shape=(32, 4), name='input_1'),
                nn.Input(shape=(32, 5), name='input_2'),
            ])
            self.assertIs(output, nn.get_tensor('Concat/output'))

    def test_bn(self):
        """Compnents consisting BatchNormalization layer are retrieved"""
        base_scope, scope = self.get_scope(), 'BN'
        with nn.variable_scope(base_scope) as vs:
            input_ = nn.Input(shape=(32, 4), name='input')
            layer = nn.fetch_layer('BatchNormalization')(scope=scope)
            output = layer(input_)
            mean = layer.get_parameter_variable('mean')
            var = layer.get_parameter_variable('var')
            scale = layer.get_parameter_variable('scale')
            offset = layer.get_parameter_variable('offset')
            updates = layer.get_update_operations()

        with nn.variable_scope(vs, reuse=True):
            self.assertIs(mean, nn.get_variable('BN/mean'))
            self.assertIs(var, nn.get_variable('BN/var'))
            self.assertIs(scale, nn.get_variable('BN/scale'))
            self.assertIs(offset, nn.get_variable('BN/offset'))
            self.assertIs(output, nn.get_tensor('BN/output'))
            self.assertIs(updates[0], nn.get_operation('BN/update_mean'))
            self.assertIs(updates[1], nn.get_operation('BN/update_var'))
