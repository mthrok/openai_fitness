from __future__ import absolute_import

import tensorflow as tf

from luchador.common import is_iteratable
from ..base import (
    make_optimizer,
    get_optimizer,
    Optimizer,
)
from .scope import get_variable
from .wrapper import (
    Variable,
    Operation,
)

__all__ = [
    'BaseOptimizer', 'make_optimizer', 'get_optimizer',
    'SGD', 'RMSProp', 'GravesRMSProp', 'NeonRMSProp', 'AdamOptimizer',
]


class BaseOptimizer(Optimizer):
    def _parse_kwargs(self, kwargs):
        keys_and_defaults1 = [
            ('gate_gradients', 1),
            ('aggregation_method', None),
            ('colocate_gradients_with_ops', False),
            ('grad_loss', None)
        ]
        keys_and_defaults2 = [
            ('global_step', None),
            ('name', None)
        ]
        kws_compute_gradients = {
            key: kwargs.get(key, default_value)
            for key, default_value in keys_and_defaults1}
        kws_apply_gradients = {
            key: kwargs.get(key, default_value)
            for key, default_value in keys_and_defaults2}
        return [kws_compute_gradients, kws_apply_gradients]

    def minimize(self, loss, wrt=None, **kwargs):
        kws1, kws2 = self._parse_kwargs(kwargs)
        grads_and_vars = self.compute_gradients(loss, wrt=wrt, **kws1)
        return self.apply_gradients(grads_and_vars, **kws2)

    def compute_gradients(self, loss, wrt, **kwargs):
        loss = loss.get()
        if wrt is not None and not is_iteratable(wrt):
            wrt = [wrt]
        var_list = [v.get() for v in wrt] if wrt else None
        grads_and_vars = self.optimizer.compute_gradients(
            loss, var_list=var_list, **kwargs)
        return grads_and_vars

    def apply_gradients(self, grads_and_vars, **kwargs):
        minimize_op = self.optimizer.apply_gradients(grads_and_vars, **kwargs)
        self._register_slot(grads_and_vars)
        return Operation(minimize_op)

    def _register_slot(self, grads_and_vars):
        """Store TF-native optimizer slots to luchador Optimizer slots"""
        for _, var in grads_and_vars:
            for slot_name in self.optimizer.get_slot_names():
                slot = self.optimizer.get_slot(var, slot_name)
                name = '{}/{}/{}'.format(
                    self.args['name'],  var.op.name, slot_name)
                self.slot.append(Variable(slot, name=name))

    def _create_slot_var(self, var, slot_name):
        """Create slot variable for the given Variable and store it"""
        name = '{}/{}/{}'.format(
            self.args['name'], var.name.split(':')[0], slot_name)
        slot_var = get_variable(
            name=name, shape=var.get_shape(), dtype=var.dtype,
            initializer=tf.constant_initializer(0))
        self.slot.append(slot_var)
        return slot_var


class SGD(BaseOptimizer):
    def __init__(self, learning_rate, name='SGD', **kwargs):
        super(SGD, self).__init__(learning_rate=learning_rate, name=name)
        self.optimizer = tf.train.GradientDescentOptimizer(
            learning_rate, name=name, **kwargs)


class RMSProp(BaseOptimizer):
    def __init__(self, learning_rate,
                 decay=0.95, momentum=0.0,
                 epsilon=1e-2, name='RMSProp', **kwargs):
        super(RMSProp, self).__init__(
            learning_rate=learning_rate,
            decay=decay, momentum=momentum, epsilon=epsilon, name=name)
        self.optimizer = tf.train.RMSPropOptimizer(
            learning_rate, decay=decay, momentum=momentum,
            epsilon=epsilon, **kwargs)


class NeonRMSProp(BaseOptimizer):
    def __init__(self, learning_rate, decay=0.95, epsilon=1e-6,
                 name='NeonRMSProp', **kwargs):
        super(NeonRMSProp, self).__init__(
            learning_rate=learning_rate,
            decay=decay, epsilon=epsilon, name=name)
        self.optimizer = tf.train.GradientDescentOptimizer(
            learning_rate, name=name)
        self.decay = decay
        self.epsilon = epsilon

    def apply_gradients(self, grads_and_vars, **kwargs):
        rms_updates = []
        new_grads_and_vars = []
        args = self.args
        decay, ep = args['decay'], args['epsilon']
        for grad, var in grads_and_vars:
            rms = self._create_slot_var(var, 'rms').get()

            new_rms = rms + (1. - decay) * (tf.square(grad) - rms)
            new_grad = tf.truediv(grad, tf.sqrt(new_rms + ep) + ep)

            rms_updates.append(rms.assign(new_rms))
            new_grads_and_vars.append((new_grad, var))
        train_op = self.optimizer.apply_gradients(new_grads_and_vars)
        updates = [train_op] + rms_updates
        return Operation(tf.group(*updates))


class GravesRMSProp(BaseOptimizer):
    def __init__(self, learning_rate,
                 decay1=0.0, decay2=0.95, epsilon=1e-2,
                 name='GravesRMSProp', **kwargs):
        super(GravesRMSProp, self).__init__(
            learning_rate=learning_rate,
            decay1=decay1, decay2=decay2, epsilon=epsilon, name=name)
        self.decay1 = decay1
        self.decay2 = decay2
        self.epsilon = epsilon
        self.optimizer = tf.train.GradientDescentOptimizer(
            learning_rate, name=name)

    def apply_gradients(self, grads_and_vars, **kwargs):
        updates, new_grads_and_vars = [], []
        args = self.args
        d1, d2, ep = args['decay1'], args['decay2'], args['epsilon']
        for grad, var in grads_and_vars:
            mean_grad1 = self._create_slot_var(var, 'grad_mean').get()
            mean_grad2 = self._create_slot_var(var, 'grad_squared_mean').get()

            new_mean_grad1 = d1 * mean_grad1 + (1.0 - d1) * grad
            new_mean_grad2 = d2 * mean_grad2 + (1.0 - d2) * tf.square(grad)

            rms = tf.sqrt(new_mean_grad2 - tf.square(new_mean_grad1) + ep)
            new_grad = tf.truediv(grad, rms)

            updates.append(mean_grad1.assign(new_mean_grad1))
            updates.append(mean_grad2.assign(new_mean_grad2))
            new_grads_and_vars.append((new_grad, var))

        updates.append(self.optimizer.apply_gradients(new_grads_and_vars))
        return Operation(tf.group(*updates))


class AdamOptimizer(BaseOptimizer):
    def __init__(self, learning_rate,
                 beta1=0.9, beta2=0.999,
                 epsilon=1e-08, name='Adam', **kwargs):
        super(AdamOptimizer, self).__init__(
            learning_rate=learning_rate,
            beta1=beta1, beta2=beta2, epsilon=epsilon, name=name)
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate, beta1=beta1, beta2=beta2,
            epsilon=epsilon, **kwargs)

    def apply_gradients(self, grads_and_vars, **kwargs):
        ret = super(AdamOptimizer, self).apply_gradients(
            grads_and_vars, **kwargs)
        name = '{}/{}'.format(self.args['name'], 'beta1_power')
        self.slot.append(Variable(self.optimizer._beta1_power, name=name))
        name = '{}/{}'.format(self.args['name'], 'beta2_power')
        self.slot.append(Variable(self.optimizer._beta2_power, name=name))
        return ret
