"""Module to render CartPole"""
from __future__ import absolute_import

import numpy as np

import luchador.util.render


class CartPoleRenderer(object):
    """Provide CartPole visualization"""
    def __init__(self, env):
        self._env = env

        self._renderer = None
        self._cart = None
        self._cart_trans = None
        self._pole = None
        self._pole_trans = None

    def init(self):
        """Initialize window and objects"""
        self._init_window()
        self._init_grid()
        self._init_cart()
        self._init_pole()

    def _init_window(self):
        self._renderer = luchador.util.render.Renderer(width=600, height=600)
        self._renderer.init_window(color=(0.2, 0.2, 0.2, 1.0))
        self._renderer.window.on_close = self._stop_rendering

    def _stop_rendering(self):
        self._env.display_screen = False
        self._renderer.window.close()

    def _init_grid(self):
        lim = self._env.distance_limit
        self._renderer.set_transform(scale=(1./lim, 1./lim))
        color = (.2, .3, .2)
        for i in np.arange(0.0, lim, 0.2):
            for sign in [-1.0, 1.0]:
                axis = luchador.util.render.Line(
                    start=(-lim, sign*i), end=(lim, sign*i), color=color)
                self._renderer.add_geometry(axis)
                axis = luchador.util.render.Line(
                    start=(sign*i, -lim), end=(sign*i, lim), color=color)
                self._renderer.add_geometry(axis)
        color = (.2, .4, .2)
        for i in np.arange(0.0, lim, 1.0):
            for sign in [-1.0, 1.0]:
                axis = luchador.util.render.Line(
                    start=(-lim, sign*i), end=(lim, sign*i), color=color)
                self._renderer.add_geometry(axis)
                axis = luchador.util.render.Line(
                    start=(sign*i, -lim), end=(sign*i, lim), color=color)
                self._renderer.add_geometry(axis)

    def _init_cart(self):
        cart_height = 0.1 * np.sqrt(self._env.state.cart_mass)
        cart_width = 2.5 * cart_height

        right, left = cart_width / 2.0, -cart_width / 2.0
        top, bottom = cart_height / 2.0, -cart_height / 2.0
        self._cart = luchador.util.render.Polygon(
            [(left, bottom), (left, top), (right, top), (right, bottom)])
        self._cart.set_color(.5, .5, .5)
        self._cart_trans = luchador.util.render.Transform()
        self._cart.add_attr(self._cart_trans)
        self._renderer.add_geometry(self._cart)

    def _init_pole(self):
        pole_width = 0.05
        pole_height = self._env.state.pole_length

        right, left = pole_width / 2.0, -pole_width / 2.0
        top, bottom = pole_height, 0.0
        self._pole = luchador.util.render.Polygon(
            [(left, bottom), (left, top), (right, top), (right, bottom)])
        self._pole.set_color(.8, .6, .4)
        self._pole_trans = luchador.util.render.Transform()
        self._pole.add_attr(self._pole_trans)
        self._pole.add_attr(self._cart_trans)
        self._renderer.add_geometry(self._pole)

    def render(self):
        """Render the current CartPole state"""
        self._cart_trans.set_translation(self._env.state.x, 0.0)
        self._pole_trans.set_rotation(-self._env.state.theta)
        self._renderer.render()
