"""Test theano get_scope mechanism"""
from __future__ import absolute_import

import unittest

import luchador
from luchador import nn
import luchador.nn.core.base.scope.theano as be

from tests.unit.fixture import TestCase


# pylint: disable=protected-access, invalid-name
class _ScopeTestCase(TestCase):
    longMessage = True

    def tearDown(self):
        # After each test, scope should be reset to root
        expected, found = '', be._get_scope()
        # Sanitize global scope for the next test in case something went wrong
        be._reset()
        self.assertEqual(
            expected, found,
            'Variable scope was not properly closed in the last test. ')

    def _check_scope(self, expected, found=None):
        found = found or be._get_scope()
        self.assertEqual(
            expected, found,
            'Failed to update current variable scope. '
        )

    def _check_reuse(self, expected, found=None):
        found = found or be._get_flag()
        self.assertEqual(
            expected, found,
            'Failed to update current reuse flag. '
        )


@unittest.skipUnless(luchador.get_nn_backend() == 'theano', 'Theano backend')
class TestVariableScopeClass(_ScopeTestCase):
    """Test if VariableScope correctly change global flags"""
    def test_update_scope(self):
        """VariableScope updates current variable scope"""
        scopes = [self.get_scope(suffix) for suffix in ['aaa', 'bbb', 'ccc']]
        with nn.VariableScope(reuse=False, name=scopes[0]):
            self._check_scope(scopes[0])
            with nn.VariableScope(reuse=False, name=scopes[1]):
                self._check_scope(scopes[1])
                with nn.VariableScope(reuse=False, name=scopes[2]):
                    self._check_scope(scopes[2])

    def test_update_reuse_flag(self):
        """VariableScope updates current reuse flag"""
        reuse = False
        with nn.VariableScope(reuse=reuse, name=self.get_scope()):
            self._check_reuse(reuse)
            reuse = False
            with nn.VariableScope(reuse=reuse, name='scope'):
                self._check_reuse(reuse)
            reuse = True
            with nn.VariableScope(reuse=reuse, name='scope'):
                self._check_reuse(reuse)

        reuse = True
        with nn.VariableScope(reuse=reuse, name=self.get_scope()):
            self._check_reuse(reuse)
            reuse = False
            with nn.VariableScope(reuse=reuse, name='scope'):
                self._check_reuse(reuse)
                reuse = True
            with nn.VariableScope(reuse=reuse, name='scope'):
                self._check_reuse(reuse)

    def test_reusing_VS_object_correctly_reset_scope(self):
        """Reusing VariableScope changes scope and flag correctly"""
        scope, reuse = self.get_scope(), False
        with nn.VariableScope(reuse=reuse, name=scope) as vs:
            pass

        with nn.VariableScope(True, 'scope_to_be_ignored'):
            with vs:
                self._check_scope(scope)
                self._check_reuse(reuse)

    def test_VariableScope_correctly_closes_scope(self):
        """VariableScope revert current variable scope after close"""
        scopes = [self.get_scope(scope) for scope in ['aaa', 'bbb', 'ccc']]
        for scope in scopes:
            expected_scope = be._get_scope()
            expected_reuse = be._get_flag()
            with nn.VariableScope(False, scope):
                pass
            self._check_scope(expected_scope)
            self._check_reuse(expected_reuse)

    def test_reuse_variables_enables_flag(self):
        """reuse_variables call enable reuse flag"""
        base_scope = self.get_scope('base')
        with nn.VariableScope(reuse=False, name=base_scope):
            with nn.VariableScope(reuse=False, name='top') as vs:
                vs.reuse_variables()
                self._check_reuse(True)

    def test_flag_reset_after_reuse_variables(self):
        """reuse_variables is not in effect after close"""
        base_scope = self.get_scope('base')
        reuse = False
        with nn.VariableScope(reuse=reuse, name=base_scope):
            with nn.VariableScope(reuse=False, name='top1') as vs:
                vs.reuse_variables()
            self._check_reuse(reuse)

        reuse = True
        with nn.VariableScope(reuse=reuse, name=base_scope):
            with nn.VariableScope(reuse=False, name='top2') as vs:
                vs.reuse_variables()
            self._check_reuse(reuse)


@unittest.skipUnless(luchador.get_nn_backend() == 'theano', 'Theano backend')
class TestVariableScopeFuncs(_ScopeTestCase):
    """Test if variable_scope funciton correctly change global flags"""
    def test_variable_scope(self):
        """variable_scope stacks new scope on the current scope"""
        scopes = [self.get_scope('aaa'), 'bbb', 'ccc']
        with nn.variable_scope(scopes[0]):
            self._check_scope(scopes[0])
            with nn.variable_scope(scopes[1]):
                self._check_scope('/'.join(scopes[:2]))
                with nn.variable_scope(scopes[2]):
                    self._check_scope('/'.join(scopes[:3]))
            with nn.variable_scope(scopes[2]):
                self._check_scope('/'.join([scopes[0], scopes[2]]))

    def test_get_variable_scope(self):
        """get_variable_scope retrieves the current scope and reuse flag"""
        scope, reuse = self.get_scope(), False
        with nn.variable_scope(scope, reuse=reuse):
            vs = be.get_variable_scope()
            self._check_scope(expected=scope, found=vs.name)
            self._check_reuse(expected=reuse, found=vs.reuse)

        reuse = True
        with nn.variable_scope(scope, reuse=reuse):
            vs = be.get_variable_scope()
            self._check_scope(expected=scope, found=vs.name)
            self._check_reuse(expected=reuse, found=vs.reuse)

    def test_get_variable_scope_stack(self):
        """get_variable_scope retrieves the current scope and reuse flag"""
        scopes, reuses = [self.get_scope(), 'aaa'], [False, False]
        with nn.variable_scope(scopes[0], reuse=reuses[0]):
            with nn.variable_scope(scopes[1], reuse=reuses[1]):
                vs = be.get_variable_scope()
                self._check_scope(expected='/'.join(scopes), found=vs.name)
                self._check_reuse(expected=reuses[1], found=vs.reuse)

        scopes, reuses = [self.get_scope(), 'bbb'], [False, True]
        with nn.variable_scope(scopes[0], reuse=reuses[0]):
            with nn.variable_scope(scopes[1], reuse=reuses[1]):
                vs = be.get_variable_scope()
                self._check_scope(expected='/'.join(scopes), found=vs.name)
                self._check_reuse(expected=reuses[1], found=vs.reuse)

        scopes, reuses = [self.get_scope(), 'ccc'], [True, False]
        with nn.variable_scope(scopes[0], reuse=reuses[0]):
            with nn.variable_scope(scopes[1], reuse=reuses[1]):
                vs = be.get_variable_scope()
                self._check_scope(expected='/'.join(scopes), found=vs.name)
                self._check_reuse(expected=reuses[1], found=vs.reuse)

        scopes, reuses = [self.get_scope(), 'ddd'], [True, True]
        with nn.variable_scope(scopes[0], reuse=reuses[0]):
            with nn.variable_scope(scopes[1], reuse=reuses[1]):
                vs = be.get_variable_scope()
                self._check_scope(expected='/'.join(scopes), found=vs.name)
                self._check_reuse(expected=reuses[1], found=vs.reuse)


@unittest.skipUnless(luchador.get_nn_backend() == 'theano', 'Theano backend')
class TestGetVariable(_ScopeTestCase):
    """Test if get_variable correctly retrieve/create Variable"""
    def test_get_variable_reuse_variable(self):
        """get_variable create variable"""
        scope = self.get_scope()
        var1 = nn.make_variable(scope, shape=[3, 1])
        be._set_flag(True)
        var2 = nn.get_variable(scope)
        self.assertIs(
            var1.unwrap(), var2.unwrap(),
            'Reused variable should be identical to the original variable'
        )

    def test_get_variable_raises_when_reuseing_non_existent_variable(self):
        """get_variable raise when trying to reuse non existent variable"""
        be._set_flag(True)
        try:
            nn.get_variable('non_existing_variable_name')
        except ValueError:
            pass
        else:
            self.fail(
                'get_variable should raise when '
                'trying to reuse non existent variable.'
            )

    def test_make_variable_raises_when_creating_existing_variable(self):
        """make_variable raise when trying to create existent variable"""
        scope = self.get_scope()
        nn.make_variable(scope, shape=[3, 1])
        try:
            nn.make_variable(scope, shape=[3, 1])
        except ValueError:
            pass
        else:
            self.fail(
                'make_variable should raise when '
                'trying to create variable already exists.'
            )
