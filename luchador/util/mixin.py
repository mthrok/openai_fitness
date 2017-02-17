"""Defines mixins used acroos the submodules in Luchador"""
from __future__ import absolute_import

from .yaml_util import pprint_dict

__all__ = ['StoreMixin']

# pylint: disable=too-few-public-methods


class StoreMixin(object):
    """Provide common pattern to validate and store constructor arguments.

    ``StoreMixin`` implements the follwoing methods.

    .. automethod:: _store_args

    .. automethod:: _validate_args

    Examples
    --------
    >>> class Foo(StoreMixin):
    >>>     def __init__(self, number, string):
    >>>         self._store_args(numer=number, string=string)
    >>>
    >>>     def _validate_args(number, string):
    >>>         if not isinstance(number, float):
    >>>             raise TypeError('Argument `number` must be float type.')
    >>>         if not isinstance(string, str):
    >>>             raise TypeError('Argument `string` must be string type.')
    >>>
    >>> foo = Foo(0.5, 'bar')
    >>> foo.args['number'] == 0.5
    True
    >>> foo.args['string'] == 'bar'
    True
    >>> Foo('bar', 1)
    TypeError: Argument `number` must be float type.

    See Also
    --------
    CompareMixin : Base mixin which adds equality comparison to StoreMixin
    SerializeMixin : Subclass mixin which serialize object with arguments
    """
    def _store_args(self, **args):
        """Store the given arguments to ``args`` attribute after validation"""
        self._validate_args(**args)
        self.args = args

    def _validate_args(self, **args):
        """Validate the given arguments

        Subclassses inheriting this mixin can add custom validation scheme by
        overriding this method.
        """
        pass

    def __repr__(self):
        return repr({'name': self.__class__.__name__, 'args':  self.args})

    def __str__(self):
        return pprint_dict({self.__class__.__name__: self.args})
