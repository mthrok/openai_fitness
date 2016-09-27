from __future__ import absolute_import

from collections import OrderedDict

import theano
import theano.tensor as T

from luchador.common import is_iteratable
from ..base import (
    make_optimizer,
    get_optimizer,
    Optimizer,
)
from .scope import get_variable
from .initializer import Constant
from .wrapper import Operation

__all__ = [
    'BaseOptimizer', 'make_optimizer', 'get_optimizer',
    'SGD', 'RMSProp', 'GravesRMSProp', 'NeonRMSProp', 'AdamOptimizer',
]


class BaseOptimizer(Optimizer):
    def minimize(self, loss, wrt, **kwargs):
        grads_and_vars = self.compute_gradients(loss, wrt, **kwargs)
        return self.apply_gradients(grads_and_vars)

    def compute_gradients(self, loss, wrt, **kwargs):
        if not is_iteratable(wrt):
            wrt = [wrt]
        loss, wrt = loss.get(), [v.get() for v in wrt]
        grads = theano.grad(loss, wrt)
        return [(grad, var) for grad, var in zip(grads, wrt)]

    def get_parameter_variables(self):
        return self.slot

    def _create_slot_var(self, var, slot_name):
        """Create slot variable for the given Variable and store it"""
        value = var.get_value(borrow=True)
        name = '{}/{}/{}'.format(
            self.args['name'], var.name.split(':')[0], slot_name)
        slot_var = get_variable(
            name=name, shape=value.shape, dtype=value.dtype,
            initializer=Constant(0), broadcastable=var.broadcastable)
        self.slot.append(slot_var)
        return slot_var

    def _create_slot(self, initial_value, slot_name):
        """Create scalar slot variable common to variables"""
        name = '{}/{}'.format(self.args['name'], slot_name)
        slot_var = get_variable(
            name=name, shape=[],
            initializer=Constant(initial_value), broadcastable=True)
        self.slot.append(slot_var)
        return slot_var


class SGD(BaseOptimizer):
    def __init__(self, learning_rate, name='SGD', **kwargs):
        super(SGD, self).__init__(
            learning_rate=learning_rate, name=name)

    def apply_gradients(self, grads_and_vars):
        updates = OrderedDict()
        for grad, var in grads_and_vars:
            updates[var] = var - self.args['learning_rate'] * grad
        return Operation(op=updates)


class RMSProp(BaseOptimizer):
    def __init__(self, learning_rate, decay=0.95, momentum=0.0,
                 epsilon=1e-2, name='RMSProp', **kwargs):
        super(RMSProp, self).__init__(
            learning_rate=learning_rate,
            decay=decay, momentum=momentum,
            epsilon=epsilon, name=name)

    def apply_gradients(self, grads_and_vars):
        updates = OrderedDict()
        args = self.args
        decay, momentum = args['decay'], args['momentum']
        ep, lr = args['epsilon'], args['learning_rate']
        for grad, var in grads_and_vars:
            mom = self._create_slot_var(var, 'momentum').get()
            rms = self._create_slot_var(var, 'rms').get()

            new_rms = rms + (1.0 - decay) * (T.square(grad) - rms)
            new_mom = mom * momentum + lr * grad / (T.sqrt(new_rms + ep))
            new_var = var - new_mom

            updates[rms] = new_rms
            updates[mom] = new_mom
            updates[var] = new_var
        return Operation(op=updates)


class NeonRMSProp(BaseOptimizer):
    def __init__(self, learning_rate, decay=0.95, epsilon=1e-6,
                 name='NeonRMSProp', **kwargs):
        super(NeonRMSProp, self).__init__(
            learning_rate=learning_rate,
            decay=decay, epsilon=epsilon, name=name)

    def apply_gradients(self, grads_and_vars):
        updates = OrderedDict()
        args = self.args
        decay, ep, lr = args['decay'], args['epsilon'], args['learning_rate']
        for grad, var in grads_and_vars:
            rms = self._create_slot_var(var, 'rms').get()

            new_rms = rms + (1.0 - decay) * (T.square(grad) - rms)
            new_var = var - lr * grad / (T.sqrt(new_rms + ep) + ep)

            updates[rms] = new_rms
            updates[var] = new_var
        return Operation(op=updates)


class GravesRMSProp(BaseOptimizer):
    """RMSProp used in DQN paper[1] and described in A.Graves paper [2]

    [1] https://github.com/kuz/DeepMind-Atari-Deep-Q-Learner/blob/4b9f5a79b03ea0cfc512ed1c11f1b00bc875bc57/dqn/NeuralQLearner.lua#L265  # nopep8
    [2] http://arxiv.org/pdf/1308.0850v5.pdf
    """
    def __init__(self, learning_rate,
                 decay1=0.95, decay2=0.95,
                 epsilon=1e-2, name='GravesRMSProp'):
        super(GravesRMSProp, self).__init__(
            learning_rate=learning_rate,
            decay1=decay1, decay2=decay2, epsilon=epsilon, name=name)

    def apply_gradients(self, grads_and_vars):
        updates = OrderedDict()
        args = self.args
        d1, d2 = args['decay1'], args['decay2']
        ep, lr = args['epsilon'], args['learning_rate']
        for grad, var in grads_and_vars:
            mean_grad1 = self._create_slot_var(var, 'grad_mean').get()
            mean_grad2 = self._create_slot_var(var, 'grad_squared_mean').get()

            new_mean_grad1 = d1 * mean_grad1 + (1.0 - d1) * grad
            new_mean_grad2 = d2 * mean_grad2 + (1.0 - d2) * T.square(grad)

            rms = T.sqrt(new_mean_grad2 - T.square(new_mean_grad1) + ep)
            new_grad = grad / rms

            delta_var = -lr * new_grad
            new_var = var + delta_var

            updates[mean_grad1] = new_mean_grad1
            updates[mean_grad2] = new_mean_grad2
            updates[var] = new_var
        return Operation(op=updates)


class AdamOptimizer(BaseOptimizer):
    def __init__(self, learning_rate,
                 beta1=0.9, beta2=0.999,
                 epsilon=1e-08, name='Adam', **kwargs):
        super(AdamOptimizer, self).__init__(
            learning_rate=learning_rate,
            beta1=beta1, beta2=beta2, epsilon=epsilon, name=name)

    def apply_gradients(self, grads_and_vars):
        args = self.args
        beta1, beta2 = args['beta1'], args['beta2']
        ep, lr = args['epsilon'], args['learning_rate']
        updates = OrderedDict()

        beta1_power = self._create_slot(beta1, 'beta1_power').get()
        beta2_power = self._create_slot(beta2, 'beta2_power').get()

        new_beta1_power = beta1_power * beta1
        new_beta2_power = beta2_power * beta2

        alpha = lr * T.sqrt(1.0 - beta2_power) / (1.0 - beta1_power)

        updates[beta1_power] = new_beta1_power
        updates[beta2_power] = new_beta2_power

        for grad, var in grads_and_vars:
            m = self._create_slot_var(var, 'm').get()
            v = self._create_slot_var(var, 'v').get()

            new_m = m + (1.0 - beta1) * (grad - m)
            new_v = v + (1.0 - beta2) * (T.square(grad) - v)
            new_var = var - (new_m * alpha) / (T.sqrt(new_v) + ep)

            updates[m] = new_m
            updates[v] = new_v
            updates[var] = new_var
        return Operation(op=updates)
