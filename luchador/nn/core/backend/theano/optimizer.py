"""Implement optimizers in Theano backend"""
from __future__ import absolute_import

from collections import OrderedDict

import theano
import theano.tensor as T

import luchador.util
from luchador.nn.core.base import get_initializer
from . import wrapper

__all__ = [
    'SGD', 'RMSProp', 'NeonRMSProp', 'GravesRMSProp', 'Adam', 'Adamax'
]
# pylint: disable=invalid-name,too-many-locals,too-few-public-methods,no-member


class OptimizerMixin(object):
    """Adds Theano-specific helper methods to base Optimizer"""
    def _run_backend_specific_init(self):
        pass

    def _minimize(self, loss, wrt, **kwargs):
        grads_and_vars = self.compute_gradients(loss, wrt, **kwargs)
        return self.apply_gradients(grads_and_vars)

    @staticmethod
    def _compute_gradients(loss, wrt, **kwargs):
        """Compute gradient

        Parameters
        ----------
        loss : Tensor
            loss to be minimized

        wrt : Variable or list of Variables
            Term for which loss Tensor is differentiated.

        kwargs
            Other arguments passed to ``theano.gradient.grad``

        Returns
        -------
        list
            List of (gradient, variable) pairs
        """
        wrt = wrt if luchador.util.is_iteratable(wrt) else [wrt]
        wrt_ = [v.unwrap() for v in wrt if v.trainable]

        if not wrt_:
            raise ValueError('No variables to optimize.')

        # So as to match the behavior to that of Tensorflow, we return None
        # for disconnected inputs
        grads = theano.grad(
            loss.unwrap(), wrt_, disconnected_inputs='warn',
            return_disconnected='None', **kwargs)
        ret, i = [], 0
        for var in wrt:
            tensor = None
            if var.trainable:
                grad = grads[i]
                i += 1
                if grad is not None:
                    name_ = '{}_grad'.format(var.name)
                    tensor = wrapper.Tensor(grad, shape=var.shape, name=name_)
            ret.append((tensor, var))
        return ret

    def _create_slot_var(self, var, name):
        """Create slot variable for the given Variable

        Typical usage is to create variables to hold moving average
        of the given Variable

        Parameters
        ----------
        var : theano.SharedVariable
            Variable of which size and dtype are used to create slot.

        name : str
            The name of slot.

        Returns
        -------
        Variable
            Wrapped Variable of the resulting slot variable.
        """
        value = var.get_value(borrow=True)
        var_name = var.name.split(':')[0]

        name_ = '/'.join([self.args['name'], var_name, name])
        var = wrapper.make_variable(
            name=name_, shape=value.shape, dtype=value.dtype,
            initializer=get_initializer('ConstantInitializer')(0),
            broadcastable=var.broadcastable)

        name_ = '/'.join([var_name, name])
        self._create_parameter_slot(name_, var, train=False, serialize=True)
        return var.unwrap()

    def _create_slot(self, initial_value, name):
        """Create slot variable independant to gradients and parameters

        Example use is beta parameters in Adam and Adamax optimizer.
        Only scalar type is supported.

        Parameters
        ----------
        initial_value : number
            Initial value of the resulting slot

        name : str
            The name of slot.

        Returns
        -------
        Variable
            Wrapped Variable of the resulting slot variable.
        """
        name_ = '{}/{}'.format(self.args['name'], name)
        var = wrapper.make_variable(
            name=name_, shape=[], broadcastable=True,
            initializer=get_initializer('ConstantInitializer')(initial_value))

        self._create_parameter_slot(name, var, train=False, serialize=True)
        return var.unwrap()


class SGD(OptimizerMixin):
    """Implement RMSProp in Theano backend.

    See :any:`BaseRMSProp` for detail.
    """
    def _apply_gradients(self, grads_and_vars):
        updates = OrderedDict()
        for grad, var in grads_and_vars:
            updates[var] = var - self.args['learning_rate'] * grad
        return wrapper.Operation(op=updates)


class RMSProp(OptimizerMixin):
    """Implement RMSProp in Theano backend.

    See :any:`BaseRMSProp` for detail.
    """
    def _apply_gradients(self, grads_and_vars):
        decay, momentum = self.args['decay'], self.args['momentum']
        ep, lr = self.args['epsilon'], self.args['learning_rate']

        updates = OrderedDict()
        for grad, var in grads_and_vars:
            mom = self._create_slot_var(var, 'momentum')
            rms = self._create_slot_var(var, 'rms')

            new_rms = rms + (1.0 - decay) * (T.square(grad) - rms)
            new_mom = mom * momentum + lr * grad / (T.sqrt(new_rms + ep))
            new_var = var - new_mom

            updates[rms] = new_rms
            updates[mom] = new_mom
            updates[var] = new_var
        return wrapper.Operation(op=updates)


class NeonRMSProp(OptimizerMixin):
    """Implement NeonRMSProp in Theano backend.

    See :any:`BaseNeonRMSProp` for detail.
    """
    def _apply_gradients(self, grads_and_vars):
        decay, ep = self.args['decay'], self.args['epsilon']
        lr = self.args['learning_rate']

        updates = OrderedDict()
        for grad, var in grads_and_vars:
            rms = self._create_slot_var(var, 'rms')

            new_rms = rms + (1.0 - decay) * (T.square(grad) - rms)
            new_var = var - lr * grad / (T.sqrt(new_rms + ep) + ep)

            updates[rms] = new_rms
            updates[var] = new_var
        return wrapper.Operation(op=updates)


class GravesRMSProp(OptimizerMixin):
    """Implement GravesRMSProp in Theano backend.

    See :any:`BaseGravesRMSProp` for detail.
    """
    def _apply_gradients(self, grads_and_vars):
        d1, d2 = self.args['decay1'], self.args['decay2']
        ep, lr = self.args['epsilon'], self.args['learning_rate']

        updates = OrderedDict()
        for grad, var in grads_and_vars:
            mean_g1 = self._create_slot_var(var, 'grad_mean')
            mean_g2 = self._create_slot_var(var, 'grad_squared_mean')

            new_mean_g1 = d1 * mean_g1 + (1.0 - d1) * grad
            new_mean_g2 = d2 * mean_g2 + (1.0 - d2) * T.square(grad)

            rms = T.sqrt(new_mean_g2 - T.square(new_mean_g1) + ep)
            new_grad = grad / rms

            delta_var = -lr * new_grad
            new_var = var + delta_var

            updates[mean_g1] = new_mean_g1
            updates[mean_g2] = new_mean_g2
            updates[var] = new_var
        return wrapper.Operation(op=updates)


class Adam(OptimizerMixin):
    """Implement Adam in Theano backend.

    See :any:`BaseAdam` for detail.
    """
    def _apply_gradients(self, grads_and_vars):
        b1, b2 = self.args['beta1'], self.args['beta2']
        ep, lr = self.args['epsilon'], self.args['learning_rate']

        b1_pow = self._create_slot(b1, 'beta1_power')
        b2_pow = self._create_slot(b2, 'beta2_power')
        alpha = lr * T.sqrt(1.0 - b2_pow) / (1.0 - b1_pow)

        updates = OrderedDict()
        for grad, var in grads_and_vars:
            m = self._create_slot_var(var, 'm')
            v = self._create_slot_var(var, 'v')

            new_m = m + (1.0 - b1) * (grad - m)
            new_v = v + (1.0 - b2) * (T.square(grad) - v)
            new_var = var - (new_m * alpha) / (T.sqrt(new_v) + ep)

            updates[m] = new_m
            updates[v] = new_v
            updates[var] = new_var

        updates[b1_pow] = b1_pow * b1
        updates[b2_pow] = b2_pow * b2
        return wrapper.Operation(op=updates)


class Adamax(OptimizerMixin):
    """Implement Adamax in Theano backend.

    See :any:`BaseAdamax` for detail.
    """
    def _apply_gradients(self, grads_and_vars):
        b1, b2 = self.args['beta1'], self.args['beta2']
        ep, lr = self.args['epsilon'], self.args['learning_rate']

        b1_pow = self._create_slot(b1, 'beta1_power')
        alpha = lr / (1.0 - b1_pow)

        updates = OrderedDict()
        for grad, var in grads_and_vars:
            m = self._create_slot_var(var, 'm')
            u = self._create_slot_var(var, 'u')

            new_m = m + (1.0 - b1) * (grad - m)
            new_u = T.maximum(b2 * u, abs(grad))
            new_var = var - (new_m * alpha) / (new_u + ep)

            updates[m] = new_m
            updates[u] = new_u
            updates[var] = new_var

        updates[b1_pow] = b1_pow * b1
        return wrapper.Operation(op=updates)
