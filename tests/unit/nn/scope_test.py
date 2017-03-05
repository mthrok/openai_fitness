"""Test Wapper methods"""
from __future__ import absolute_import

from luchador import nn

from tests.unit import fixture

# pylint: disable=invalid-name,protected-access

_VARIABLES = nn.core.base.wrapper.store._VARIABLES


class TestVariableStore(fixture.TestCase):
    """Test Variable/Tensor store mechanism"""
    def test_get_variable_creates_variable(self):
        """get_variable create variable"""
        scope, var_name = self.get_scope(), 'foo'
        full_name = '/'.join([scope, var_name])

        self.assertTrue(full_name not in _VARIABLES)
        with nn.variable_scope(scope, reuse=True):
            with self.assertRaises(ValueError):
                nn.get_variable(var_name)

        with nn.variable_scope(scope, reuse=False):
            variable = nn.make_variable(var_name, shape=[3, 1])
        self.assertTrue(full_name in _VARIABLES)

        self.assertIs(variable, _VARIABLES[full_name])
        with nn.variable_scope(scope, reuse=True):
            self.assertIs(variable, nn.get_variable(var_name))

    def test_get_tensor_from_current_scope(self):
        """get_tensor retrieve existing tensor"""
        scope, name = self.get_scope(), 'foo'
        with nn.variable_scope(scope):
            tensor = fixture.create_ones_tensor([3, 1], 'float32', name=name)
            self.assertIs(tensor, nn.get_tensor(name))

        self.assertIs(tensor, nn.get_tensor('{}/{}'.format(scope, name)))

        with self.assertRaises(ValueError):
            nn.get_tensor(name)

    def test_get_input_from_current_scope(self):
        """get_input retrieve existing input"""
        scope, name = self.get_scope(), 'foo'
        with nn.variable_scope(scope):
            op = nn.Input(shape=[], name=name)
            self.assertIs(op, nn.get_input(name))

        self.assertIs(op, nn.get_input('{}/{}'.format(scope, name)))

        with self.assertRaises(ValueError):
            nn.get_input(name)

    def test_get_operation_from_current_scope(self):
        """get_operation retrieve existing operation"""
        scope, name = self.get_scope(), 'foo'
        with nn.variable_scope(scope):
            op = nn.Operation(op=None, name=name)
            self.assertIs(op, nn.get_operation(name))

        self.assertIs(op, nn.get_operation('{}/{}'.format(scope, name)))

        with self.assertRaises(ValueError):
            nn.get_operation(name)
