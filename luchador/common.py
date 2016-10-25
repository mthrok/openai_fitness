"""Defines methods and mixins used acroos the submodules in Luchador"""

from __future__ import absolute_import

import yaml


def pprint_dict(dictionary):
    """Pretty-print dictionary in YAML style"""
    return yaml.dump(dictionary, default_flow_style=False)


def is_iteratable(obj):
    """Return True if given object is iteratable"""
    try:
        list(obj)
        return True
    except TypeError:
        return False


###############################################################################
# Getter mechanism
def get_subclasses(class_):
    """Get the list of all subclasses

    This function is intended to use in ``get_*`` functions such as
    ``get_agent``, ``get_env``, ``get_layer``, ``get_optimizer``,
    ``get_initializer`` etc. for flexible access.

    Parameters
    ----------
    class_ : type
        Class of which subclasses are searched

    Returns
    -------
    list
        List of subclasses of the given class
    """
    ret = []
    for subclass in class_.__subclasses__():
        # As we adopt subclassing-base-interface approach to realize backend
        # switch, we prefer to return subclasses in the order similar to
        # python's inheritence resolution.
        # So the descendant classes comes before their anscestors.
        ret.extend(get_subclasses(subclass))
        ret.append(subclass)
    return ret


###############################################################################
# Mixins
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


class CompareMixin(StoreMixin):
    """Extend StoreMixin by providing method to compare ``args``

    .. automethod:: __eq__

    .. automethod:: __ne__

    Examples
    --------
    >>> class Foo(CompareMixin):
    >>>     def __init__(self, arg):
    >>>         self._store_args(arg=arg)
    >>>
    >>> foo1 = Foo('bar')
    >>> foo2 = Foo('bar')
    >>> foo3 = Foo('foo')
    >>> foo1 == foo2
    True
    >>> foo1 == foo3
    False

    See Also
    --------
    StoreMixin : Base mixin which stores costructor arguments
    SerializeMixin : Subclass mixin which serialize object with arguments
    """
    def __eq__(self, other):
        """Checks if the other object has `same configuration` with this object

        `Same configuration` here means objects are of the same (or subclassed)
        type and have the same ``args`` attribute value.
        """
        if isinstance(other, self.__class__):
            return self.args == other.args
        return NotImplemented

    def __ne__(self, other):
        """Checks if the other object is not equal to this object"""
        return not self.__eq__(other)


class SerializeMixin(CompareMixin):
    """Extend StoreMixin to be serializable in JSON format

    Examples
    --------
    >>> class Foo(SerializeMixin):
    >>>     def __init__(self, number, string):
    >>>         self._store_args(numer=number, string=string)
    >>>
    >>> foo = Foo(0.5, 'bar')
    >>> print(foo.serialize())
    {'number': 0.5, 'string': 'bar'}

    See Also
    --------
    StoreMixin : Base mixin which stores costructor arguments
    CompareMixin : Base mixin which adds equality comparison to StoreMixin
    """
    def serialize(self):
        """Serialize object configuration (constructor arguments)

        Returns
        -------
        dict
           Arguments stored via ::func:`_store_args` method.
        """
        args = {}
        for key, val in self.args.items():
            args[key] = val.serialize() if hasattr(val, 'serialize') else val
        return {
            'name': self.__class__.__name__,
            'args': args
        }
