from __future__ import absolute_import

import unittest

import numpy as np

from luchador import nn


def _create_variables(shape=(3, 4)):
    init = nn.initializer.Constant
    with nn.variable_scope('source'):
        src = nn.get_variable('source', shape=shape, initializer=init(value=1))
    with nn.variable_scope('target'):
        tgt = nn.get_variable('taget', shape=shape, initializer=init(value=0))
    return src, tgt


class TestMisc(unittest.TestCase):
    def test_sync_without_tau(self):
        """sync op copies values from source variables to target variables"""
        with nn.variable_scope(self.id().replace('.', '/')):
            source_var, target_var = _create_variables()
            sync_op = nn.build_sync_op([source_var], [target_var], tau=None)

        session = nn.Session()
        session.initialize()

        src_val, tgt_val = session.run([source_var, target_var])
        self.assertTrue((src_val == 1).all())
        self.assertTrue((tgt_val == 0).all())

        session.run(updates=sync_op)

        src_val, tgt_val = session.run([source_var, target_var])
        self.assertTrue((src_val == 1).all())
        self.assertTrue((tgt_val == src_val).all())

    def test_sync_with_tau(self):
        """sync op copies weighted sum of source and target variables"""
        tau = 0.1
        with nn.variable_scope(self.id().replace('.', '/')):
            source_var, target_var = _create_variables()
            sync_op = nn.build_sync_op([source_var], [target_var], tau=tau)

        session = nn.Session()
        session.initialize()

        src_val, tgt_val = session.run([source_var, target_var])
        self.assertTrue((src_val == 1).all())
        self.assertTrue((tgt_val == 0).all())

        for _ in range(10):
            expected = tau * src_val + (1 - tau) * tgt_val
            session.run(updates=sync_op)
            src_val, found = session.run([source_var, target_var])
            self.assertTrue((src_val == 1).all())
            self.assertTrue(
                np.square(expected - found).sum() < 1e-10,
                '\nExpected: \n{}\nFound: \n{}'.format(expected, found)
            )
            tgt_val = found
