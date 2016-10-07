from __future__ import absolute_import


class Component(object):
    def __init__(self, w, h, x=0, y=0, vx=0, vy=0):
        self.w = w
        self.h = h
        self.x = self.x0 = x
        self.y = self.y0 = y
        self.vx = self.vx0 = vx
        self.vy = self.vy0 = vy

    def reset(self):
        self.x = self.x0
        self.y = self.y0
        self.vx = self.vx0
        self.vy = self.vy0

    def update(self):
        self.x += self.vx
        self.y += self.vy

    @property
    def r(self):
        return self.x + self.w

    @property
    def b(self):
        return self.y + self.h

    @property
    def cx(self):
        return self.x + self.w / 2

    @property
    def cy(self):
        return self.y + self.h / 2


class Background(Component):
    def __init__(self, w, h):
        super(Background, self).__init__(w, h, 0, 0, 0, 0)


class Ground(Component):
    def __init__(self, w, h, y, vx, shift):
        super(Ground, self).__init__(w, h, 0, y, vx, 0)
        self.shift = shift

    def update(self):
        super(Ground, self).update()
        self.x = self.x % self.shift


class Pipe(Component):
    def __init__(self, w, h, x, y, vx):
        super(Pipe, self).__init__(w, h, x, y, vx, 0)


class Pipes(object):
    def __init__(
            self, w, h, vx, y_min, y_max, y_gap, x_gap, n_pipes, rng):
        self.pipe_w = w
        self.pipe_h = h
        self.pipe_vx = vx
        self.y_min = y_min
        self.y_max = y_max
        self.y_gap = y_gap
        self.x_gap = x_gap
        self.n_pipes = n_pipes
        self.rng = rng

    def reset(self):
        self.pipes = []
        for _ in range(self.n_pipes):
            self.add_pipe()

    def add_pipe(self):
        w, h, vx, x_gap = self.pipe_w, self.pipe_h, self.pipe_vx, self.x_gap
        x = x_gap + (self.pipes[-1][0].x if self.pipes else 2 * x_gap)
        y_t = self.rng.randint(self.y_min, self.y_max)
        y_b = y_t + h + self.y_gap
        self.pipes.append((Pipe(w, h, x, y_t, vx), Pipe(w, h, x, y_b, vx)))

    def update(self):
        for top, bottom in self.pipes:
            top.update()
            bottom.update()
        if self.pipes[0][0].r <= 0:
            self.pipes.pop(0)
            self.add_pipe()


class Player(Component):
    def __init__(self, w, h, x, y, y_max, vy, vy_flap, vy_max, ay):
        super(Player, self).__init__(w, h, x, y, 0, vy)
        self.ay = 1
        self.y_max = y_max
        self.vy_max = vy_max
        self.vy_flap = vy_flap

    def update(self, tapped):
        flapped = False
        if tapped and self.y > 0:
                self.vy = self.vy_flap
                flapped = True
        else:
            self.vy += self.ay
            self.vy = min(self.vy, self.vy_max)

        super(Player, self).update()
        self.y = min(self.y, self.y_max)
        return flapped
