from __future__ import absolute_import

import yaml


def pprint_dict(dictionary):
    return yaml.dump(dictionary, default_flow_style=False)


def is_iteratable(l):
    try:
        list(l)
        return True
    except Exception:
        return False


###############################################################################
# Getter mechanism
def get_subclasses(Class):
    """Get the list of all subclasses

    This function is intended to use in get_* functions such as
    get_agent, get_env, get_layer, get_optimizer, get_initializer ...
    for flexible access.
    """
    ret = []
    for SubClass in Class.__subclasses__():
        # As we adopt subclassing-base-interface approach to realize backend
        # switch, we prefer to return subclasses in the order similar to
        # python's inheritence resolution.
        # So the descendant classes comes before their anscestors.
        ret.extend(get_subclasses(SubClass))
        ret.append(SubClass)
    return ret


###############################################################################
# Mixins
class StoreMixin(object):
    def _store_args(self, **args):
        """Store initializer arguments to `args` attribute
        Args:
            args (dict): Arguments passed to __init__.
            This will be used to recreate a subclass instance.
        """
        self._validate_args(args)
        self.args = args

    def _validate_args(self, args):
        """Validate arguments
        Args:
            args (dict): Arguments passed to __init__.
        """
        pass

    def __repr__(self):
        return repr({'name': self.__class__.__name__, 'args':  self.args})


class CompareMixin(StoreMixin):
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.args == other.args
        return NotImplemented

    def __ne__(self, other):
        return not self.__eq__(other)


class SerializeMixin(CompareMixin):
    def serialize(self):
        """Serialize object configuration (constructor arguments)"""
        args = {}
        for key, val in self.args.items():
            args[key] = val.serialize() if hasattr(val, 'serialize') else val
        return {
            'name': self.__class__.__name__,
            'args': args
        }
