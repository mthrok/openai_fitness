"""Defines methods and mixins used acroos the submodules in Luchador"""
from __future__ import absolute_import

__all__ = ['is_iteratable', 'get_subclasses']


def is_iteratable(obj):
    """Return True if given object is iteratable"""
    try:
        list(obj)
        return True
    except TypeError:
        return False


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
