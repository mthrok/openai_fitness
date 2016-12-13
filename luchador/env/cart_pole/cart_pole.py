"""CartPole controll environment

Taken from https://webdocs.cs.ualberta.ca/~sutton/book/code/pole.c
"""
from __future__ import division
from __future__ import absolute_import

import numpy as np

from ..base import BaseEnvironment, Outcome
from . import render

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
            self.renderer = None
            self.pole = None
            self.cart = None
            self.pole_trans = None
            self.cart_trans = None
            self._init_renderer()

    @property
    def n_actions(self):
        return 2

    def _get_observation(self):
        return {
            'x': self.state.x,
            'x_dot': self.state.x_dot,
            'theta': self.state.theta,
            'theta_dot': self.state.theta_dot,
        }

    def reset(self):
        """Reset environment state."""
        self.state.reset()
        return Outcome(
            reward=0.0, terminal=False, observation=self._get_observation())

    def step(self, action):
        self.state.update(10 if action else -10)

        terminal = (
            abs(self.state.x) > self.distance_limit or
            abs(self.state.theta) > _deg2rad(self.angle_limit)
        )
        reward = -1.0 if terminal else 0.0

        if self.display_screen:
            self._update_screen()

        return Outcome(reward=reward, terminal=terminal,
                       observation=self._get_observation())

    ###########################################################################
    # Rendering
    def _stop_rendering(self):
        self.display_screen = False

    def _init_renderer(self):
        self.renderer = render.Renderer(width=600, height=600)
        self.renderer.init_window(color=(0.2, 0.2, 0.2, 1.0))
        self.renderer.window.on_close = self._stop_rendering

        self._init_grid()
        self._init_cart()
        self._init_pole()

    def _init_grid(self):
        lim = self.distance_limit
        self.renderer.set_transform(scale=(1./lim, 1./lim))
        color = (.2, .3, .2)
        for i in np.arange(0.0, lim, 0.2):
            for sign in [-1.0, 1.0]:
                axis = render.Line(
                    start=(-lim, sign*i), end=(lim, sign*i), color=color)
                self.renderer.add_geometry(axis)
                axis = render.Line(
                    start=(sign*i, -lim), end=(sign*i, lim), color=color)
                self.renderer.add_geometry(axis)
        color = (.2, .4, .2)
        for i in np.arange(0.0, lim, 1.0):
            for sign in [-1.0, 1.0]:
                axis = render.Line(
                    start=(-lim, sign*i), end=(lim, sign*i), color=color)
                self.renderer.add_geometry(axis)
                axis = render.Line(
                    start=(sign*i, -lim), end=(sign*i, lim), color=color)
                self.renderer.add_geometry(axis)

    def _init_cart(self):
        cart_height = 0.1 * np.sqrt(self.state.cart_mass)
        cart_width = 2.5 * cart_height

        right, left = cart_width / 2.0, -cart_width / 2.0
        top, bottom = cart_height / 2.0, -cart_height / 2.0
        self.cart = render.Polygon(
            [(left, bottom), (left, top), (right, top), (right, bottom)])
        self.cart.set_color(.5, .5, .5)
        self.cart_trans = render.Transform()
        self.cart.add_attr(self.cart_trans)
        self.renderer.add_geometry(self.cart)

    def _init_pole(self):
        pole_width = 0.05
        pole_height = self.state.pole_length

        right, left = pole_width / 2.0, -pole_width / 2.0
        top, bottom = pole_height, 0.0
        self.pole = render.Polygon(
            [(left, bottom), (left, top), (right, top), (right, bottom)])
        self.pole.set_color(.8, .6, .4)
        self.pole_trans = render.Transform()
        self.pole.add_attr(self.pole_trans)
        self.pole.add_attr(self.cart_trans)
        self.renderer.add_geometry(self.pole)

    def _update_screen(self):
        self.cart_trans.set_translation(self.state.x, 0.0)
        self.pole_trans.set_rotation(-self.state.theta)
        self.renderer.render()
