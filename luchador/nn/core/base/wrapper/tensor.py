"""Define base Tensor type"""
from __future__ import absolute_import

from ..scope import get_variable_scope
from .store import register, retrieve
from .wrapper import BaseWrapper

__all__ = ['BaseTensor', 'get_tensor']


class BaseTensor(BaseWrapper):
    """Base wrapper class for Tensor"""
    def __init__(self, tensor, shape, name, dtype):
        super(BaseTensor, self).__init__(
            tensor=tensor, shape=shape, name=name, dtype=dtype)
        if name:
            self.register(name)

    def register(self, name):
        """Register this Tensor to global list of Tensors for later re-use

        Parameters
        ----------
        name : str
            Name to store this Tensor object.

        Note
        ----
        This method is provided for reusing Tensor which was created without
        a given name. Tensors created with proper name is automatically
        registered, thus there is no need to register.

        An exmaple of Tensor without name is those created with overloaded
        operator, such as __add__.

        >>> tensor2 = tensor1 + 0.1

        To retrieve `tensor2` with `get_tensor` method, you need to call
        tensor2.register().
        """
        register('tensor', name, self)


def get_tensor(name):
    """Get an instance of ``Tensor`` from the current or the global scope

    Parameters
    ----------
    name : str
        name of ``Tensor`` instance to get

    Returns
    -------
    Tensor
    """
    try:
        scope = get_variable_scope().name
        name_ = '{}/{}'.format(scope, name) if scope else name
        return retrieve('tensor', name_)
    except ValueError:
        pass
    return retrieve('tensor', name)


def get_grad(var):
    """Get gradient ``Tensor`` corresponding to the given ``Variable``

    In optimizers, gradient tensors are registered in global scope,
    following the naming pattern ``<scope>/<variable_name>_grad``.

    This function automatically build such name from the given ``Variable``
    and the current scope name.

    To properly fetch the corresponding gradient ``Tensor``, this function
    must be called in the scope where gradient ``Tensor`` was defined.

    Examples
    --------
    >>> from luchador import nn
    >>> x = nn.get_variable(shape=(), name='x')
    >>> # Variable x is registered with name 'x'
    >>> y = x * x
    >>> sgd = nn.optimizer.SGD(learning_rate=0.1)
    >>> with nn.variable_scope('optimization'):
    >>>    sgd.minimize(loss=y, wrt=x)
    >>>    # dydx is registered with name '/optimization/x_grad'
    >>>    dydx2 = nn.get_grad_tensor(x)
    >>>    assert dydx1 is dydx2

    Parameters
    ----------
    var : Variable
        ``Variable`` object of which grad is retrieved.

    Returns
    -------
    Tensor
        ``Tensor`` object which is a gradient of given ``Variable``
    """
    return get_tensor('{}_grad'.format(var.name))
