"""Simple 2D rendering with OpenGL"""
from __future__ import division
from __future__ import absolute_import

import abc

import pyglet
import pyglet.gl as gl

_RAD2DEG = 57.29577951308232


class Attribute(object):
    """Provide interface for defining attribute"""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def enable(self):
        """Enable attribute"""
        pass

    def disable(self):
        """Disable attribute"""
        pass


class Color(Attribute):
    """Color attribute"""
    def __init__(self, red, green, blue, alpha=1.0):
        self.red = red
        self.green = green
        self.blue = blue
        self.alpha = alpha

    def enable(self):
        """Apply color"""
        gl.glColor4f(self.red, self.green, self.blue, self.alpha)

    def set_color(self, red, green, blue, alpha):
        """Update color property"""
        self.red = red
        self.green = green
        self.blue = blue
        self.alpha = alpha


class LineStyle(Attribute):
    """Line style attribute"""
    def __init__(self, style):
        self.style = style

    def enable(self):
        """Apply line style"""
        gl.glEnable(gl.GL_LINE_STIPPLE)
        gl.glLineStipple(1, self.style)

    def disable(self):
        """Disable line style"""
        gl.glDisable(gl.GL_LINE_STIPPLE)


class LineWidth(Attribute):
    """Line width attribute"""
    def __init__(self, width):
        self.width = width

    def enable(self):
        """Apply line width"""
        gl.glLineWidth(self.width)


class Geometry(object):
    """Primitive object with customizable attributes"""

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self._color = Color(0., 0., 0.)
        self.attrs = [self._color]

    def render(self):
        """Render this geometry"""
        for attr in reversed(self.attrs):
            attr.enable()
        self._render()
        for attr in self.attrs:
            attr.disable()

    @abc.abstractmethod
    def _render(self):
        """Actual Rendering method"""
        pass

    def add_attr(self, attr):
        """Add attribute"""
        self.attrs.append(attr)

    def set_color(self, red, green, blue, alpha=1.):
        """Set color attribute"""
        self._color.set_color(red, green, blue, alpha)


class Point(Geometry):
    """Point"""
    def __init__(self):
        super(Point, self).__init__()

    def _render(self):
        gl.glBegin(gl.GL_POINTS)
        gl.glVertex3f(0., 0., 0.)
        gl.glEnd()


class Line(Geometry):
    """Sinple line segment"""
    def __init__(self, start, end, color=None):
        super(Line, self).__init__()
        self.start = start
        self.end = end

        self._linewidth = LineWidth(1)
        self.add_attr(self._linewidth)
        self._color = color or (0.0, 0.0, 0.0)

    def _render(self):
        gl.glBegin(gl.GL_LINES)
        gl.glColor3f(*self._color)
        gl.glVertex2f(*self.start)
        gl.glVertex2f(*self.end)
        gl.glEnd()

    def set_linewidth(self, width):
        """Set line width"""
        self._linewidth.width = width


class PolyLine(Geometry):
    """Multiple line segments"""
    def __init__(self, vertices, close):
        super(PolyLine, self).__init__()

        self.vertices = vertices
        self.close = close
        self._linewidth = LineWidth(1)
        self.add_attr(self._linewidth)

    def _render(self):
        arg = gl.GL_LINE_LOOP if self.close else gl.GL_LINE_STRIP
        gl.glBegin(arg)
        for vertex in self.vertices:
            gl.glVertex3f(vertex[0], vertex[1], 0.)
        gl.glEnd()

    def set_linewidth(self, width):
        """Set line width"""
        self._linewidth.width = width


class Polygon(Geometry):
    """Polygon"""
    def __init__(self, vertices):
        super(Polygon, self).__init__()
        self.vertices = vertices

    def _render(self):
        n_vertices = len(self.vertices)
        if n_vertices < 4:
            arg = gl.GL_TRIANGLES
        if n_vertices == 4:
            arg = gl.GL_QUADS
        else:
            arg = gl.GL_POLYGON

        gl.glBegin(arg)
        for vertex in self.vertices:
            gl.glVertex3f(vertex[0], vertex[1], 0.)
        gl.glEnd()


class Transform(Attribute):
    """Transform position, rotation and scaling of Geometry"""
    def __init__(self, translation=(0., 0.), rotation=0., scale=(1., 1.)):
        self.set_translation(*translation)
        self.set_rotation(rotation)
        self.set_scale(*scale)

    def enable(self):
        """Enable transoformation"""
        gl.glPushMatrix()
        gl.glTranslatef(*self.translation)
        gl.glRotatef(*self.rotation)
        gl.glScalef(*self.scale)

    def disable(self):
        """Disable transoformation"""
        gl.glPopMatrix()

    def set_translation(self, trans_x, trans_y):
        """Set translation"""
        self.translation = (trans_x, trans_y, 0.)

    def set_rotation(self, radian):
        """Set rotation"""
        self.rotation = (_RAD2DEG * radian, 0., 0., 1.)

    def set_scale(self, scale_x, scale_y):
        """Set scaling"""
        self.scale = (scale_x, scale_y, 1.)


class Renderer(object):
    """Manage and render Geometries"""
    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.geometries = []
        self.transform = Transform()
        self.window = None

    def set_transform(self, translation=None, rotation=None, scale=None):
        """Set global transformation"""
        if translation:
            self.transform.set_translation(*translation)
        if rotation:
            self.transform.set_rotation(*rotation)
        if scale:
            self.transform.set_scale(*scale)

    def _on_resize(self, width, _):
        ratio = self.height / self.width
        height = int(ratio * width)
        self.window.set_size(width, height)
        gl.glViewport(0, 0, width, height)

    def init_window(self, color=None):
        """Initialize window"""
        self.window = pyglet.window.Window(
            width=self.width, height=self.height, resizable=True)
        self.window.on_resize = self._on_resize

        color = color or (0, 0, 0, 1.0)

        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glClearColor(*color)

    def render(self):
        """Render objects"""
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

        self.transform.enable()
        for geom in self.geometries:
            geom.render()
        self.transform.disable()

        self.window.flip()

    def add_geometry(self, geometry):
        """Add geometry"""
        self.geometries.append(geometry)
