from __future__ import absolute_import

import unittest

import luchador
from luchador import nn

# pylint: disable=protected-access, invalid-name


@unittest.skipUnless(luchador.get_nn_backend() == 'theano', 'Theano backend')
class TestVariableScopeClass(unittest.TestCase):
    def tearDown(self):
        # After each test, scope should be reset to root
        expected, found = '', nn.scope._get_scope()
        # Sanitize global scope for the next test in case something went wrong
        nn.scope._reset()
        self.assertEqual(
            expected, found,
            'Variable scope was not properly closed in the last test. '
            'Expected: "{}", Found: "{}"'.format(expected, found))

    def test_update_scope(self):
        """VariableScope updates current variable scope"""
        scopes = ['aaa', 'bbb', 'ccc']
        with nn.scope.VariableScope(reuse=False, name=scopes[0]):
            expected = scopes[0]
            found = nn.scope._get_scope()
            self.assertEqual(
                expected, found,
                'Failed to update current variable scope. '
                'Expected: "{}", Found: "{}"'.format(expected, found)
            )
            with nn.scope.VariableScope(reuse=False, name=scopes[1]):
                expected = scopes[1]
                found = nn.scope._get_scope()
                self.assertEqual(
                    expected, found,
                    'Failed to update current variable scope. '
                    'Expected: "{}", Found: "{}"'.format(expected, found)
                )
                with nn.scope.VariableScope(reuse=False, name=scopes[2]):
                    expected = scopes[2]
                    found = nn.scope._get_scope()
                    self.assertEqual(
                        expected, found,
                        'Failed to update current variable scope. '
                        'Expected: "{}", Found: "{}"'.format(expected, found)
                    )

    def test_update_reuse_flag(self):
        """VariableScope updates current reuse flag"""
        with nn.scope.VariableScope(reuse=False, name='scope'):
            expected = False
            found = nn.scope._get_flag()
            self.assertEqual(
                expected, found,
                'Failed to update current reuse flag. '
                'Expected: "{}", Found: "{}"'.format(expected, found)
            )
            with nn.scope.VariableScope(reuse=False, name='scope'):
                expected = False
                found = nn.scope._get_flag()
                self.assertEqual(
                    expected, found,
                    'Failed to update current reuse flag. '
                    'Expected: "{}", Found: "{}"'.format(expected, found)
                )
            with nn.scope.VariableScope(reuse=True, name='scope'):
                expected = True
                found = nn.scope._get_flag()
                self.assertEqual(
                    expected, found,
                    'Failed to update current reuse flag. '
                    'Expected: "{}", Found: "{}"'.format(expected, found)
                )
        with nn.scope.VariableScope(reuse=True, name='scope'):
            expected = True
            found = nn.scope._get_flag()
            self.assertEqual(
                expected, found,
                'Failed to update current reuse flag. '
                'Expected: "{}", Found: "{}"'.format(expected, found)
            )
            with nn.scope.VariableScope(reuse=False, name='scope'):
                expected = False
                found = nn.scope._get_flag()
                self.assertEqual(
                    expected, found,
                    'Failed to update current reuse flag. '
                    'Expected: "{}", Found: "{}"'.format(expected, found)
                )
            with nn.scope.VariableScope(reuse=True, name='scope'):
                expected = True
                found = nn.scope._get_flag()
                self.assertEqual(
                    expected, found,
                    'Failed to update current reuse flag. '
                    'Expected: "{}", Found: "{}"'.format(expected, found)
                )

    def test_reusing_VS_object_correctly_reset_scope(self):
        """Reusing VariableScope changes scope and flag correctly"""
        scope, flag = 'aaa', False
        with nn.scope.VariableScope(reuse=flag, name=scope) as vs:
            pass

        with nn.scope.VariableScope(True, 'scope_to_be_ignored'):
            with vs:
                expected = scope
                found = nn.scope._get_scope()
                self.assertEqual(
                    expected, found,
                    'Failed to update current variable scope '
                    'when reusing VariableScope. '
                    'Expected: "{}", Found: "{}"'.format(expected, found)
                )
                expected = flag
                found = nn.scope._get_flag()
                self.assertEqual(
                    expected, found,
                    'Failed to update current reuse flag '
                    'when reusing VariableScope. '
                    'Expected: "{}", Found: "{}"'.format(expected, found)
                )

    def test_VariableScope_correctly_closes_scope(self):
        """VariableScope revert current variable scope after close"""
        scopes = ['aaa', 'bbb', 'ccc']
        for scope in scopes:
            pre_scope = nn.scope._get_scope()
            pre_reuse = nn.scope._get_flag()
            with nn.scope.VariableScope(False, scope):
                pass
            post_scope = nn.scope._get_scope()
            post_reuse = nn.scope._get_flag()

            expected = pre_scope
            found = post_scope
            self.assertEqual(
                expected, found,
                'Failed to reset variable scope after closing a scope.'
                'Expected: "{}", Found: "{}"'.format(expected, found)
            )

            expected = pre_reuse
            found = post_reuse
            self.assertEqual(
                expected, found,
                'Failed to reset reuse flag after closing a scope.'
                'Expected: "{}", Found: "{}"'.format(expected, found)
            )

    def test_reuse_variables_enables_flag(self):
        """reuse_variables call enable reuse flag"""
        with nn.scope.VariableScope(reuse=False, name='base_scope'):
            with nn.scope.VariableScope(reuse=False, name='base') as vs:
                vs.reuse_variables()
                expected = True
                found = nn.scope._get_flag()
                self.assertEqual(
                    expected, found,
                    'Failed to update current variable scope '
                    'with call to reuse_variables. '
                    'Expected: "{}", Found: "{}"'.format(expected, found)
                )

    def test_flag_reset_after_reuse_variables(self):
        """reuse_variables call enable reuse flag"""
        with nn.scope.VariableScope(reuse=False, name='base_scope'):
            with nn.scope.VariableScope(reuse=False, name='base') as vs:
                vs.reuse_variables()
            expected = False
            found = nn.scope._get_flag()
            self.assertEqual(
                expected, found,
                'Failed to reset the flag after calling reuse_variables'
                'Expected: "{}", Found: "{}"'.format(expected, found)
            )

        with nn.scope.VariableScope(reuse=True, name='base_scope'):
            with nn.scope.VariableScope(reuse=False, name='base') as vs:
                vs.reuse_variables()
            expected = True
            found = nn.scope._get_flag()
            self.assertEqual(
                expected, found,
                'Failed to reset the flag after calling reuse_variables'
                'Expected: "{}", Found: "{}"'.format(expected, found)
            )


@unittest.skipUnless(luchador.get_nn_backend() == 'theano', 'Theano backend')
class TestVariableScopeFuncs(unittest.TestCase):
    def tearDown(self):
        # After each test, scope should be reset to root
        expected, found = '', nn.scope._get_scope()
        # Sanitize global scope for the next test in case something went wrong
        nn.scope._reset()
        self.assertEqual(
            expected, found,
            'Variable scope was not properly closed in the last test. '
            'Expected: "{}", Found: "{}"'.format(expected, found))

    def test_variable_scope(self):
        """variable_scope stacks new scope on the current scope"""
        scopes = ['aaa', 'bbb', 'ccc']
        with nn.scope.variable_scope(scopes[0]):
            expected = scopes[0]
            found = nn.scope._get_scope()
            self.assertEqual(
                expected, found,
                'Failed to update current variable scope with variable_scope'
                'Expected: "{}", Found: "{}"'.format(expected, found)
            )
            with nn.scope.variable_scope(scopes[1]):
                expected = '/'.join(scopes[:2])
                found = nn.scope._get_scope()
                self.assertEqual(
                    expected, found,
                    'Failed to stack variable scopes with variable_scope'
                    'Expected: "{}", Found: "{}"'.format(expected, found)
                )
                with nn.scope.variable_scope(scopes[2]):
                    expected = '/'.join(scopes[:3])
                    found = nn.scope._get_scope()
                    self.assertEqual(
                        expected, found,
                        'Failed to stack variable scopes with variable_scope'
                        'Expected: "{}", Found: "{}"'.format(expected, found)
                    )
            with nn.scope.variable_scope(scopes[2]):
                expected = '/'.join([scopes[0], scopes[2]])
                found = nn.scope._get_scope()
                self.assertEqual(
                    expected, found,
                    'Failed to stack variable scopes with variable_scope'
                    'Expected: "{}", Found: "{}"'.format(expected, found)
                )

    def test_get_variable_scope(self):
        """get_variable_scope retrieves the current scope and reuse flag"""
        scope, reuse = 'aaa', False
        with nn.scope.variable_scope(scope, reuse=reuse):
            vs = nn.scope.get_variable_scope()

            expected = scope
            found = vs.name
            self.assertEqual(
                expected, found,
                'Failed to get the current scope with get_variable_scope'
                'Expected: "{}", Found: "{}"'.format(expected, found)
            )

            expected = reuse
            found = vs.reuse
            self.assertEqual(
                expected, found,
                'Failed to get the current reuse flag with get_variable_scope'
                'Expected: "{}", Found: "{}"'.format(expected, found)
            )

        scope, reuse = 'aaa', True
        with nn.scope.variable_scope(scope, reuse=reuse):
            vs = nn.scope.get_variable_scope()

            expected = scope
            found = vs.name
            self.assertEqual(
                expected, found,
                'Failed to get the current scope with get_variable_scope'
                'Expected: "{}", Found: "{}"'.format(expected, found)
            )

            expected = reuse
            found = vs.reuse
            self.assertEqual(
                expected, found,
                'Failed to get the current reuse flag with get_variable_scope'
                'Expected: "{}", Found: "{}"'.format(expected, found)
            )

        scopes, reuses = ['aaa', 'bbb'], [False, False]
        with nn.scope.variable_scope(scopes[0], reuse=reuses[0]):
            with nn.scope.variable_scope(scopes[1], reuse=reuses[1]):
                vs = nn.scope.get_variable_scope()

                expected = '/'.join(scopes)
                found = vs.name
                self.assertEqual(
                    expected, found,
                    'Failed to get the current scope with get_variable_scope'
                    'Expected: "{}", Found: "{}"'.format(expected, found)
                )

                expected = reuses[1]
                found = vs.reuse
                self.assertEqual(
                    expected, found,
                    'Failed to get the current reuse flag '
                    'with get_variable_scope'
                    'Expected: "{}", Found: "{}"'.format(expected, found)
                )

        scopes, reuses = ['aaa', 'bbb'], [False, True]
        with nn.scope.variable_scope(scopes[0], reuse=reuses[0]):
            with nn.scope.variable_scope(scopes[1], reuse=reuses[1]):
                vs = nn.scope.get_variable_scope()

                expected = '/'.join(scopes)
                found = vs.name
                self.assertEqual(
                    expected, found,
                    'Failed to get the current scope with get_variable_scope'
                    'Expected: "{}", Found: "{}"'.format(expected, found)
                )

                expected = reuses[1]
                found = vs.reuse
                self.assertEqual(
                    expected, found,
                    'Failed to get the current reuse flag '
                    'with get_variable_scope'
                    'Expected: "{}", Found: "{}"'.format(expected, found)
                )

        scopes, reuses = ['aaa', 'bbb'], [True, True]
        with nn.scope.variable_scope(scopes[0], reuse=reuses[0]):
            with nn.scope.variable_scope(scopes[1], reuse=reuses[1]):
                vs = nn.scope.get_variable_scope()

                expected = '/'.join(scopes)
                found = vs.name
                self.assertEqual(
                    expected, found,
                    'Failed to get the current scope with get_variable_scope'
                    'Expected: "{}", Found: "{}"'.format(expected, found)
                )

                expected = reuses[1]
                found = vs.reuse
                self.assertEqual(
                    expected, found,
                    'Failed to get the current reuse flag '
                    'with get_variable_scope'
                    'Expected: "{}", Found: "{}"'.format(expected, found)
                )

        scopes, reuses = ['aaa', 'bbb'], [True, False]
        with nn.scope.variable_scope(scopes[0], reuse=reuses[0]):
            with nn.scope.variable_scope(scopes[1], reuse=reuses[1]):
                vs = nn.scope.get_variable_scope()

                expected = '/'.join(scopes)
                found = vs.name
                self.assertEqual(
                    expected, found,
                    'Failed to get the current scope with get_variable_scope'
                    'Expected: "{}", Found: "{}"'.format(expected, found)
                )

                expected = reuses[1]
                found = vs.reuse
                self.assertEqual(
                    expected, found,
                    'Failed to get the current reuse flag '
                    'with get_variable_scope'
                    'Expected: "{}", Found: "{}"'.format(expected, found)
                )


@unittest.skipUnless(luchador.get_nn_backend() == 'theano', 'Theano backend')
class TestGetVariable(unittest.TestCase):
    def tearDown(self):
        # After each test, scope should be reset to root
        expected, found = '', nn.scope._get_scope()
        # Sanitize global scope for the next test in case something went wrong
        nn.scope._reset()
        self.assertEqual(
            expected, found,
            'Variable scope was not properly closed in the last test. '
            'Expected: "{}", Found: "{}"'.format(expected, found))

    def test_get_variable_creates_variable(self):
        """get_variable create variable"""
        name = 'test_var'
        self.assertTrue(name not in nn.wrapper._VARIABLES)
        nn.scope.get_variable('test_var', shape=[3, 1])
        self.assertTrue(name in nn.wrapper._VARIABLES)

    def test_get_variable_reuse_variable(self):
        """get_variable create variable"""
        name = 'test_var'
        var1 = nn.scope.get_variable(name, shape=[3, 1])
        nn.scope._set_flag(True)
        var2 = nn.scope.get_variable(name)
        self.assertIs(
            var1.unwrap(), var2.unwrap(),
            'Reused variable should be identical to the original variable')

    def test_get_variable_raises_when_reuseing_non_existent_variable(self):
        """get_variable raise when trying to reuse non existent variable"""
        nn.scope._set_flag(True)
        try:
            nn.scope.get_variable('non_existent_var')
        except ValueError:
            pass
        else:
            self.fail('get_variable should raise when '
                      'trying to reuse non existent variable.')

    def test_get_variable_raises_when_creating_already_existing_variable(self):
        """get_variable raise when trying to create existent variable"""
        name = 'aaa'
        nn.scope.get_variable(name, shape=[3, 1])
        try:
            nn.scope.get_variable(name)
        except ValueError:
            pass
        else:
            self.fail('get_variable should raise when '
                      'trying to create variable already exists.')
