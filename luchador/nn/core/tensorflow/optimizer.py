from __future__ import absolute_import

import tensorflow as tf

from luchador.common import is_iteratable
from ..base import (
    make_optimizer,
    get_optimizer,
    BaseOptimizer,
    BaseSGD,
    BaseRMSProp,
    BaseNeonRMSProp,
    BaseGravesRMSProp,
    BaseAdam,
    BaseAdamax,
)
from .scope import get_variable
from .initializer import Constant
from .wrapper import (
    Variable,
    Operation,
)

__all__ = [
    'BaseOptimizer', 'make_optimizer', 'get_optimizer',
    'SGD',
    'RMSProp', 'GravesRMSProp', 'NeonRMSProp',
    'Adam', 'Adamax',
]


class TFOptimizerMixin(object):
    """Adds TF-specific helper methods to base Optimizer"""
    def init(self):
        """Initialize underlying TF optimizer to SGD

        Manually implemented optimizers use SGD as the actual step for
        updating the paramters after modifyin gradients.

        So for convenience, this mixin initialize underlying optimizer with SGD
        """
        self.optimizer = tf.train.GradientDescentOptimizer(
            self.args['learning_rate'], name=self.args['name'])

    def minimize(self, loss, wrt=None, **kwargs):
        """Create minimization op which updates parameter variables

        Args:
          loss (Tensor): Loss Tensor to be minimized

          wrt ([list of] Variables): Variables with which loss is minimzied.

          **kwargs: Other arguments passed to either compute_gradients or
                    apply_gradients of Tenasorflow native Optimizer.

        Returns:
          Operation: Minimization operation
        """
        kws1, kws2 = self._parse_kwargs(kwargs)
        grads_and_vars = self.compute_gradients(loss, wrt=wrt, **kws1)
        return self.apply_gradients(grads_and_vars, **kws2)

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

    ###########################################################################
    def compute_gradients(self, loss, wrt, **kwargs):
        """Compute gradient of loss with respect to wrt.

        This method works in similar way as Tensorflow Optimizers'
        compute_gradient method.

        Args:
          loss (Tensor): Loss Tensor to be minimized

          wrt ([list of] Variables): Variables with which gradient is computed

          **kwargs: Other arguments passed to compute_gradients of
                    underlying Tenasorflow native Optimizer.

        Returns:
          List of Tensor pairs: Gradient and corresponding variable pairs.
            Unlike other methods, each tensor is not wrapped with Luchador's
            TensorWrapper so they are Tensorflow's native Tensor objects.
        """
        wrt = [wrt] if wrt is not None and not is_iteratable(wrt) else wrt
        var_list = [v.unwrap() for v in wrt if v.trainable] if wrt else None
        return self.optimizer.compute_gradients(
            loss=loss.unwrap(), var_list=var_list, **kwargs)

    def apply_gradients(self, grads_and_vars, **kwargs):
        """Apply grads_and_vars to optimizer to create minimization Operation

        This also store slot variables of TF native Optimizers to luchador
        Optimizer.

        Args:
          grads_and_vars (list of Tensor pairs): Value returned by
                                                 compute_gradient method.

          **kwargs: Other arguments passed to apply_gradients of
                    underlying Tenasorflow native Optimizer.

        Returns:
          Operation: Operation which updates parmeter variables.
        """
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
        """Create slot variable for the given Variable

        Typical usage is to create variables to hold moving average
        of the given Variable
        """
        name = '{}/{}/{}'.format(
            self.args['name'], var.name.split(':')[0], slot_name)
        slot_var = get_variable(
            name=name, shape=var.get_shape(), dtype=var.dtype,
            initializer=tf.constant_initializer(0))
        self.slot.append(slot_var)
        return slot_var.unwrap()

    def _create_slot(self, initial_value, slot_name):
        """Create slot variable independant to gradients and parameters

        Example use is beta parameter in Adamax optimizer.
        Currently only scalar type is supported.
        """
        name = '{}/{}'.format(self.args['name'], slot_name)
        slot_var = get_variable(
            name=name, shape=[],
            initializer=Constant(initial_value))
        self.slot.append(slot_var)
        return slot_var.unwrap()


class SGD(TFOptimizerMixin, BaseSGD):
    def init(self):
        self.optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=self.args['learning_rate'], name=self.args['name'],
            use_locking=self.args.get('use_locking', False))


class RMSProp(TFOptimizerMixin, BaseRMSProp):
    def init(self):
        self.optimizer = tf.train.RMSPropOptimizer(
            learning_rate=self.args['learning_rate'],
            decay=self.args['decay'], momentum=self.args['momentum'],
            epsilon=self.args['epsilon'], name=self.args['name'],
            use_locking=self.args.get('use_locking', False))


class NeonRMSProp(TFOptimizerMixin, BaseNeonRMSProp):
    def apply_gradients(self, grads_and_vars, **kwargs):
        with tf.name_scope(self.args['name']):
            return self._apply_gradients(grads_and_vars, **kwargs)

    def _apply_gradients(self, grads_and_vars, **kwargs):
        decay, ep = self.args['decay'], self.args['epsilon']

        updates, new_grads_and_vars = [], []
        for grad, var in grads_and_vars:
            rms = self._create_slot_var(var, 'rms')

            new_rms = rms + (1. - decay) * (tf.square(grad) - rms)
            new_grad = tf.truediv(grad, tf.sqrt(new_rms + ep) + ep)

            updates.append(rms.assign(new_rms))
            new_grads_and_vars.append((new_grad, var))

        updates.append(self.optimizer.apply_gradients(new_grads_and_vars))
        return Operation(tf.group(*updates))


class GravesRMSProp(TFOptimizerMixin, BaseGravesRMSProp):
    def apply_gradients(self, grads_and_vars, **kwargs):
        with tf.name_scope(self.args['name']):
            return self._apply_gradients(grads_and_vars, **kwargs)

    def _apply_gradients(self, grads_and_vars, **kwargs):
        d1, d2 = self.args['decay1'], self.args['decay2'],
        ep = self.args['epsilon']

        updates, new_grads_and_vars = [], []
        for grad, var in grads_and_vars:
            mean_grad1 = self._create_slot_var(var, 'grad_mean')
            mean_grad2 = self._create_slot_var(var, 'grad_squared_mean')

            new_mean_grad1 = d1 * mean_grad1 + (1.0 - d1) * grad
            new_mean_grad2 = d2 * mean_grad2 + (1.0 - d2) * tf.square(grad)

            rms = tf.sqrt(new_mean_grad2 - tf.square(new_mean_grad1) + ep)
            new_grad = tf.truediv(grad, rms)

            updates.append(mean_grad1.assign(new_mean_grad1))
            updates.append(mean_grad2.assign(new_mean_grad2))
            new_grads_and_vars.append((new_grad, var))

        updates.append(self.optimizer.apply_gradients(new_grads_and_vars))
        return Operation(tf.group(*updates))


class Adam(TFOptimizerMixin, BaseAdam):
    def init(self):
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.args['learning_rate'],
            beta1=self.args['beta1'], beta2=self.args['beta2'],
            epsilon=self.args['epsilon'],
            use_locking=self.args.get('use_locking', False),
            name=self.args['name'])

    def apply_gradients(self, grads_and_vars, **kwargs):
        ret = super(Adam, self).apply_gradients(grads_and_vars, **kwargs)
        name = '{}/{}'.format(self.args['name'], 'beta1_power')
        self.slot.append(Variable(self.optimizer._beta1_power, name=name))
        name = '{}/{}'.format(self.args['name'], 'beta2_power')
        self.slot.append(Variable(self.optimizer._beta2_power, name=name))
        return ret


class Adamax(TFOptimizerMixin, BaseAdamax):
    def apply_gradients(self, grads_and_vars, **kwargs):
        with tf.name_scope(self.args['name']):
            return self._apply_gradients(grads_and_vars, **kwargs)

    def _apply_gradients(self, grads_and_vars, **kwargs):
        b1, b2 = self.args['beta1'], self.args['beta2']
        ep = self.args['epsilon']

        beta1_power = self._create_slot(b1, 'beta1_power')
        alpha = 1.0 / (1.0 - beta1_power)

        updates, new_grads_and_vars = [], []
        for grad, var in grads_and_vars:
            m = self._create_slot_var(var, 'm')
            u = self._create_slot_var(var, 'u')

            new_m = b1 * m + (1.0 - b1) * grad
            new_u = tf.maximum(b2 * u, tf.abs(grad))
            new_grad = (new_m * alpha) / (new_u + ep)

            updates.append(m.assign(new_m))
            updates.append(u.assign(new_u))
            new_grads_and_vars.append((new_grad, var))

        updates.append(beta1_power.assign(beta1_power * b1))
        updates.append(self.optimizer.apply_gradients(new_grads_and_vars))
        return Operation(tf.group(*updates))
