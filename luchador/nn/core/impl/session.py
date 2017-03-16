"""Implement Session class"""
from __future__ import absolute_import

from ..base.session import BaseSession
from ..backend import session

__all__ = ['Session', 'get_session']

_SESSIONS = {}


class Session(session.Session, BaseSession):
    """Implement Tensorflow-like Session class which executes computation"""
    def __init__(self, graph=None, config=None):
        super(Session, self).__init__(graph=graph, config=config)

    def run(self, outputs=None, inputs=None,
            updates=None, givens=None, name=None):
        """Run computation and update values

        Parameters
        ----------
        outputs : list of Tensors
            Tensors of which values are fetched

        inputs : dict
            Keys are the input Tensors required to compute values of output
            Tensors. Values are actual values to feed to Tensors.

        updates : Operation or list of Operations
            Updates variables

        givens : dict
            Same as inputs

        name : str
            Not used. Compatibility for theano backend

        Returns
        -------
        [list of] NumPy ND Arrays
            The resulting values corresponding the given `outputs` values
        """
        return self._run(
            outputs=outputs, inputs=inputs,
            updates=updates, givens=givens, name=name)

    @property
    def graph(self):
        """Returns Graph object. TF only."""
        return self._get_graph()

    def initialize(self):
        """Initialize variables. TF only.
        No effect in Theano backend as Variables are already initialized.
        """
        self._initialize()

    def close(self):
        """Finalize session"""
        self._close()

    def load_dataset(self, dataset, cast=True, strict=True):
        """Set the value of Variables with the given values

        Parameters
        ----------
        dataset : dict
            The keys are the names of Variables to be set, values are the
            NumPy arrays with which value are used.

        cast : Bool
            Not used in Tensorflow backend as it casts dtype internally.

        strict : Bool
            When True, if dataset contains a value for Variable which is not
            defined, then ValueError exception is raised. Otherwise it will
            be skipped.
        """
        self._load_dataset(dataset=dataset, cast=cast, strict=strict)
###############################################################################


def get_session(key='default'):
    """Get session"""
    return _SESSIONS[key]


_SESSIONS['default'] = Session()
