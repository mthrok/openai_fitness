from __future__ import absolute_import

from collections import OrderedDict

import theano
import theano.tensor as T

from luchador import common
from ..base import optimizer as base_opt
from . import scope, initializer, wrapper

__all__ = [
    'SGD',
    'RMSProp', 'GravesRMSProp', 'NeonRMSProp',
    'Adam', 'Adamax',
]


class TheanoOptimizerMixin(object):
    """Adds Theano-specific helper methods to base Optimizer"""
    def _run_backend_specific_init(self):
        pass

    def minimize(self, loss, wrt, **kwargs):
        """Create minimization op which updates parameter variables

        Args:
          loss (Tensor): Loss Tensor to be minimized

          wrt ([list of] Variables): Variables with which loss is minimzied

          **kwargs: Not used. This is for the consistency with TF backend.

        Returns:
          Operation: Minimization operation
        """

        grads_and_vars = self.compute_gradients(loss, wrt)
        return self.apply_gradients(grads_and_vars)

    @staticmethod
    def compute_gradients(loss, wrt, **kwargs):
        """Compute gradient of loss with respect to wrt.

        This method works in similar way as Tensorflow Optimizers'
        compute_gradient method.

        Args:
          loss (Tensor): Loss Tensor to be minimized
          wrt ([list of] Variables): Variables with which gradient is computed

        Returns:
          List of Tensor pairs: Gradient and corresponding variable pairs.
            Unlike other methods, each tensor is not wrapped with Luchador's
            TensorWrapper so they are Theano's native Variable objects.
        """
        wrt = wrt if common.is_iteratable(wrt) else [wrt]
        loss, wrt = loss.unwrap(), [v.unwrap() for v in wrt if v.trainable]
        grads = theano.grad(loss, wrt)
        return [(grad, var) for grad, var in zip(grads, wrt)]

    def _create_slot_var(self, var, slot_name):
        """Create slot variable for the given Variable

        Typical usage is to create variables to hold moving average
        of the given Variable
        """
        value = var.get_value(borrow=True)
        name = '{}/{}/{}'.format(
            self.args['name'], var.name.split(':')[0], slot_name)
        slot_var = scope.get_variable(
            name=name, shape=value.shape, dtype=value.dtype,
            initializer=initializer.Constant(0),
            broadcastable=var.broadcastable)
        self.slot.append(slot_var)
        return slot_var

    def _create_slot(self, initial_value, slot_name):
        """Create slot variable independant to gradients and parameters

        Example use is beta parameters in Adam and Adamax optimizer.
        Only scalar type is supported.
        """
        name = '{}/{}'.format(self.args['name'], slot_name)
        slot_var = scope.get_variable(
            name=name, shape=[], broadcastable=True,
            initializer=initializer.Constant(initial_value))
        self.slot.append(slot_var)
        return slot_var


class SGD(TheanoOptimizerMixin, base_opt.BaseSGD):
    def apply_gradients(self, grads_and_vars):
        updates = OrderedDict()
        for grad, var in grads_and_vars:
            updates[var] = var - self.args['learning_rate'] * grad
        return wrapper.Operation(op=updates)


class RMSProp(TheanoOptimizerMixin, base_opt.BaseRMSProp):
    def apply_gradients(self, grads_and_vars):
        decay, momentum = self.args['decay'], self.args['momentum']
        ep, lr = self.args['epsilon'], self.args['learning_rate']

        updates = OrderedDict()
        for grad, var in grads_and_vars:
            mom = self._create_slot_var(var, 'momentum').unwrap()
            rms = self._create_slot_var(var, 'rms').unwrap()

            new_rms = rms + (1.0 - decay) * (T.square(grad) - rms)
            new_mom = mom * momentum + lr * grad / (T.sqrt(new_rms + ep))
            new_var = var - new_mom

            updates[rms] = new_rms
            updates[mom] = new_mom
            updates[var] = new_var
        return wrapper.Operation(op=updates)


class NeonRMSProp(TheanoOptimizerMixin, base_opt.BaseNeonRMSProp):
    def apply_gradients(self, grads_and_vars):
        decay, ep = self.args['decay'], self.args['epsilon']
        lr = self.args['learning_rate']

        updates = OrderedDict()
        for grad, var in grads_and_vars:
            rms = self._create_slot_var(var, 'rms').unwrap()

            new_rms = rms + (1.0 - decay) * (T.square(grad) - rms)
            new_var = var - lr * grad / (T.sqrt(new_rms + ep) + ep)

            updates[rms] = new_rms
            updates[var] = new_var
        return wrapper.Operation(op=updates)


class GravesRMSProp(TheanoOptimizerMixin, base_opt.BaseGravesRMSProp):
    def apply_gradients(self, grads_and_vars):
        d1, d2 = self.args['decay1'], self.args['decay2']
        ep, lr = self.args['epsilon'], self.args['learning_rate']

        updates = OrderedDict()
        for grad, var in grads_and_vars:
            mean_g1 = self._create_slot_var(var, 'grad_mean').unwrap()
            mean_g2 = self._create_slot_var(var, 'grad_squared_mean').unwrap()

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


class Adam(TheanoOptimizerMixin, base_opt.BaseAdam):
    def apply_gradients(self, grads_and_vars):
        b1, b2 = self.args['beta1'], self.args['beta2']
        ep, lr = self.args['epsilon'], self.args['learning_rate']

        b1_pow = self._create_slot(b1, 'beta1_power').unwrap()
        b2_pow = self._create_slot(b2, 'beta2_power').unwrap()
        alpha = lr * T.sqrt(1.0 - b2_pow) / (1.0 - b1_pow)

        updates = OrderedDict()
        for grad, var in grads_and_vars:
            m = self._create_slot_var(var, 'm').unwrap()
            v = self._create_slot_var(var, 'v').unwrap()

            new_m = m + (1.0 - b1) * (grad - m)
            new_v = v + (1.0 - b2) * (T.square(grad) - v)
            new_var = var - (new_m * alpha) / (T.sqrt(new_v) + ep)

            updates[m] = new_m
            updates[v] = new_v
            updates[var] = new_var

        updates[b1_pow] = b1_pow * b1
        updates[b2_pow] = b2_pow * b2
        return wrapper.Operation(op=updates)


class Adamax(TheanoOptimizerMixin, base_opt.BaseAdamax):
    def apply_gradients(self, grads_and_vars):
        b1, b2 = self.args['beta1'], self.args['beta2']
        ep, lr = self.args['epsilon'], self.args['learning_rate']

        b1_pow = self._create_slot(b1, 'beta1_power').unwrap()
        alpha = lr / (1.0 - b1_pow)

        updates = OrderedDict()
        for grad, var in grads_and_vars:
            m = self._create_slot_var(var, 'm').unwrap()
            u = self._create_slot_var(var, 'u').unwrap()

            new_m = m + (1.0 - b1) * (grad - m)
            new_u = T.maximum(b2 * u, abs(grad))
            new_var = var - (new_m * alpha) / (new_u + ep)

            updates[m] = new_m
            updates[u] = new_u
            updates[var] = new_var

        updates[b1_pow] = b1_pow * b1
        return wrapper.Operation(op=updates)
