import numpy as np
import matplotlib.pyplot as plt  # numpy and matplotlib for visualizations
from functools import lru_cache  # cache function results to improve recursive runtime
import operations as ops


def linear_regress(x_data, y_data):
    n = len(x_data)
    # find orthogonal bases of span(x_data, 1)
    orthogonal_base_1 = ops.ones(n)
    proj = ops.project(x_data, orthogonal_base_1)
    orthogonal_base_2 = ops.sub(x_data, proj)

    # find projection of y_data onto span(x_data, 1) to find point that minimizes distance/error
    # but find it in terms of orthogonal_base_2 so we can find it in terms of a multiplier and an
    # intercept.
    slope = ops.proj_mult(y_data, orthogonal_base_2)
    intercept = ops.proj_mult(y_data, orthogonal_base_1) - proj[0] * slope  # projected onto 1, always the same for every axis
    return slope, intercept


def poly_regress(x_data, y_data, degree=1):
    xs = [ops.power(x_data, i) for i in range(degree + 1)]
    return function_regress(xs, y_data)


def function_regress(xs, y_data):
    n = len(y_data)

    orthogonal_1 = xs[0]
    orthogonal_bases = [orthogonal_1]
    slopes = ops.zeros(len(xs))
    slopes[0] = ops.proj_mult(y_data, orthogonal_1)

    xs = xs[1:]
    for i in range(len(xs)):
        proj_sum = ops.zeros(n)
        for base in orthogonal_bases:
            proj_sum = ops.add(proj_sum, ops.project(xs[i], base))

        orthogonal_base = ops.sub(xs[i], proj_sum)
        cur_slope = ops.proj_mult(y_data, orthogonal_base)

        # get the scale of effect of the current slope on each degree less than or equal to deg
        @lru_cache(maxsize=None)
        def get_xhat_mults(deg):
            mults = ops.zeros(deg + 1)
            mults[-1] = 1
            for j in range(deg):
                multiplier = -ops.proj_mult(xs[deg - 1], orthogonal_bases[j])
                next_xhat_mult = get_xhat_mults(j)
                for k in range(len(next_xhat_mult)):
                    mults[k] += multiplier * next_xhat_mult[k]
            return mults

        xhat_mults = get_xhat_mults(i + 1)
        for j in range(len(xhat_mults)):
            slopes[j] += cur_slope * xhat_mults[j]

        orthogonal_bases.append(orthogonal_base)

    return slopes


class PointEditor:
    def __init__(self, ax, x_init=None, y_init=None):
        self.ax = ax
        self.fig = ax.figure

        self.x = list(x_init if x_init is not None else [0, 1, 2])
        self.y = list(y_init if y_init is not None else [0, 1, 4])

        self.points = []
        for xi, yi in zip(self.x, self.y):
            p, = ax.plot([xi], [yi], 'o', ms=9, picker=5)
            self.points.append(p)

        self.fit_line, = ax.plot([], [], '-', lw=2)

        self.drag_i = None
        self.drag_anchor = None

        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        self.update_fit()
        self.ax.set_title("Drag points. Press 'a' to add a point at the cursor.")
        self.fig.canvas.draw_idle()

    def update_fit(self):
        if len(self.x) >= 2:
            regress = function_regress([function(self.x) for function in functions], self.y)

            xmin, xmax = min(self.x), max(self.x)

            if xmin == xmax:
                xmin -= 1.0
                xmax += 1.0
            xs = np.linspace(xmin, xmax, 200)

            ys = np.full_like(xs, 0, dtype=float)
            for i in range(len(regress)):
                ys += ops.mult_scalar(functions[i](xs), regress[i])

            self.fit_line.set_data(xs, ys)
        else:
            self.fit_line.set_data([], [])

    def redraw(self):
        self.fig.canvas.draw_idle()

    def add_point(self, x_new, y_new):
        self.x.append(float(x_new))
        self.y.append(float(y_new))
        p, = self.ax.plot([x_new], [y_new], 'o', ms=9, picker=5)
        self.points.append(p)
        self.update_fit()
        self.redraw()

    def pick_point_index(self, event):
        for i, p in enumerate(self.points):
            contains, _ = p.contains(event)
            if contains:
                return i
        return None

    def on_press(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return
        i = self.pick_point_index(event)
        if i is None:
            return
        x0, y0 = self.x[i], self.y[i]
        self.drag_i = i
        self.drag_anchor = (x0, y0, event.xdata, event.ydata)

    def on_motion(self, event):
        if self.drag_i is None or event.inaxes != self.ax or event.xdata is None:
            return
        x0, y0, xpress, ypress = self.drag_anchor
        dx, dy = event.xdata - xpress, event.ydata - ypress
        nx, ny = x0 + dx, y0 + dy

        self.x[self.drag_i] = float(nx)
        self.y[self.drag_i] = float(ny)
        self.points[self.drag_i].set_xdata([nx])
        self.points[self.drag_i].set_ydata([ny])

        self.update_fit()
        self.fig.canvas.draw_idle()

    def on_release(self, event):
        if self.drag_i is None:
            return
        self.drag_i = None
        self.drag_anchor = None
        self.update_fit()
        self.redraw()

    def on_key(self, event):
        if event.key == 'a' and event.inaxes == self.ax and event.xdata is not None:
            self.add_point(event.xdata, event.ydata)


def intercept(x_data):
    return ops.ones(len(x_data))


def linear(x_data):
    return x_data


def sqrt(x_data):
    return ops.power(x_data, 1/2)


functions = [intercept, linear, ops.ln, sqrt]


if __name__ == "__main__":
    fig, ax = plt.subplots()
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.grid(True, alpha=0.3)

    x0 = [1, 2, 3, 4, 5]
    y0 = [1, 2, 3, 4, 5]

    editor = PointEditor(ax, x0, y0)
    plt.show()
