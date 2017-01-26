"""CartPole controll environment from [1]_ according to [2]_

References
----------
.. [1] A. G. Barto, R. S. Sutton, and C. W. Anderson (1983)
       Neuronlike adaptive elements that can solve difficult
       learning control problems
       IEEE Transactions on Systems, Man, and Cybernetics,
       vol. SMC-13, pp. 834-846, Sept./Oct. 1983.

.. [2] https://webdocs.cs.ualberta.ca/~sutton/book/code/pole.c

"""
from __future__ import division
from __future__ import absolute_import

import numpy as np

from .renderer import CartPoleRenderer
from ..base import BaseEnvironment, Outcome

__all__ = ['CartPole']


class _State(object):
    """Physical state of CartPole

    See :any:CartPole for constructor arguments
    """
    def __init__(self, cart_mass, pole_mass, pole_length, gravity, dt):
        self.x = 0.0
        self.x_dot = 0.0
        self.theta = 0.0
        self.theta_dot = 0.0

        self.gravity = gravity
        self.cart_mass = cart_mass
        self.pole_mass = pole_mass
        self.pole_length = pole_length
        self.dt = dt

        self.total_mass = self.pole_mass + self.cart_mass
        self.pole_mass_length = self.pole_mass * self.pole_length

    def reset(self):
        """Reset CartPole to the initial position"""
        self.x = 0.0
        self.x_dot = 0.0
        self.theta = 0.0
        self.theta_dot = 0.0

    def update(self, force):
        """Update the CartPole state for one time step

        Parameters
        ----------
        force : float
            The force applied to cart
        """
        cos = np.cos(self.theta)
        sin = np.sin(self.theta)

        tmp = self.pole_mass_length * (self.theta_dot ** 2) * sin + force
        tmp /= self.total_mass

        theta_acc = self.gravity * sin - cos * tmp
        theta_acc /= (4/3 - self.pole_mass * (cos ** 2) / self.total_mass) / 2
        x_acc = tmp - self.pole_mass_length * theta_acc * cos / self.total_mass

        self.x += self.dt * self.x_dot
        self.x_dot += self.dt * x_acc
        self.theta += self.dt * self.theta_dot
        self.theta_dot += self.dt * theta_acc


def _deg2rad(deg):
    return deg * np.pi / 180.0


class CartPole(BaseEnvironment):
    """Cart-Pole balancing problem from classic RL problem

    Parameters
    ----------
    angle_limit : float
        Upper limit of angular deviation in degree for teminal state.

    distance_limit : float
        Upper limit of distance deviation in meter for teminal state.

    cart_mass : float
        Mass of cart

    pole_mass : float
        Mass of pole

    pole_length : float
        Length of pole

    gravity : float
        Gravity constant. Default: 9.8 [m/ss]

    dt : float
        Unit time step. Default 0.2 [second]

    display_screen : Boolean
        Visualize the state when enabled
    """
    def __init__(self,
                 angle_limit=12,
                 distance_limit=2.4,
                 cart_mass=1.0,
                 pole_mass=0.1,
                 pole_length=0.1,
                 gravity=9.8,
                 dt=0.02,
                 display_screen=False):
        self.state = _State(
            cart_mass=cart_mass, pole_mass=pole_mass,
            pole_length=pole_length, gravity=gravity, dt=dt)

        self.angle_limit = angle_limit
        self.distance_limit = distance_limit

        self.display_screen = display_screen

        if self.display_screen:
            self._renderer = CartPoleRenderer(self)
            self._renderer.init()

    @property
    def n_actions(self):
        return 2

    def _get_outcome(self):
        terminal = (
            abs(self.state.x) > self.distance_limit or
            abs(self.state.theta) > _deg2rad(self.angle_limit)
        )
        return Outcome(
            reward=-1. if terminal else 0.,
            terminal=terminal,
            state={
                'x': self.state.x,
                'x_dot': self.state.x_dot,
                'theta': self.state.theta,
                'theta_dot': self.state.theta_dot,
            }
        )

    def reset(self):
        """Reset environment state."""
        self.state.reset()
        return self._get_outcome()

    def step(self, action):
        self.state.update(10 if action else -10)

        if self.display_screen:
            self._renderer.render()

        return self._get_outcome()

    def __str__(self):
        return '\n'.join([
            'CartPole Env:',
            '    angle_limit: {}'.format(self.angle_limit),
            '    distance_limit: {}'.format(self.distance_limit),
            '    cart_mass: {}'.format(self.state.cart_mass),
            '    pole_mass: {}'.format(self.state.pole_mass),
            '    pole_length: {}'.format(self.state.pole_length),
            '    gravity: {}'.format(self.state.gravity),
            '    dt: {}'.format(self.state.dt),
            '    display_screen: {}'.format(self.display_screen),
        ])
