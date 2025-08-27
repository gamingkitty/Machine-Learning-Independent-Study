import numpy as np
import matplotlib.pyplot as plt # numpy and matplotlib for visualizations
from functools import lru_cache # cache function results to improve recursive runtime


def sub(arr1, arr2):
    return [arr1[i] - arr2[i] for i in range(len(arr1))]


def add(arr1, arr2):
    return [arr1[i] + arr2[i] for i in range(len(arr1))]


def mult(arr1, arr2):
    return [arr1[i] * arr2[i] for i in range(len(arr1))]


def dot(arr1, arr2):
    return sum(arr1[i] * arr2[i] for i in range(len(arr1)))


def mult_scalar(arr, scalar):
    return [scalar * data for data in arr]


def project(point, base):
    return mult_scalar(base, proj_mult(point, base))


def proj_mult(point, base):
    return dot(point, base) / dot(base, base)


def ones(n):
    return [1 for _ in range(n)]


def zeros(n):
    return [0 for _ in range(n)]


def linear_regress(x_data, y_data):
    n = len(x_data)
    # find orthogonal bases of span(x_data, 1)
    orthogonal_base_1 = ones(n)
    proj = project(x_data, orthogonal_base_1)
    orthogonal_base_2 = sub(x_data, proj)

    # find projection of y_data onto span(x_data, 1) to find point that minimizes distance/error
    # but find it in terms of orthogonal_base_2 so we can find it in terms of a multiplier and an
    # intercept.
    slope = proj_mult(y_data, orthogonal_base_2)
    intercept = proj_mult(y_data, orthogonal_base_1) - proj[0] * slope # projected onto 1, always the same for every axis
    return slope, intercept


def poly_regress(x_data, y_data, degree=1):
    n = len(x_data)
    xs = [[(x ** (i + 1)) for x in x_data] for i in range(degree)]
    orthogonal_1 = ones(n)
    orthogonal_bases = [orthogonal_1]
    slopes = zeros(degree + 1)
    slopes[0] = proj_mult(y_data, orthogonal_1)
    for i in range(len(xs)):
        proj_sum = zeros(n)
        for base in orthogonal_bases:
            proj_sum = add(proj_sum, project(xs[i], base))

        orthogonal_base = sub(xs[i], proj_sum)
        cur_slope = proj_mult(y_data, orthogonal_base)
        # get the scale of effect of the current slope on each degree less than or equal to deg
        @lru_cache(maxsize=None)
        def get_xhat_mults(deg):
            if deg == 0:
                return [1]
            mults = zeros(deg + 1)
            mults[-1] = 1
            for j in range(deg):
                multiplier = -proj_mult(xs[deg - 1], orthogonal_bases[j])
                next_xhat_mult = get_xhat_mults(j)
                for k in range(len(next_xhat_mult)):
                    mults[k] += multiplier * next_xhat_mult[k]
            return mults

        xhat_mults = get_xhat_mults(i + 1)
        for j in range(len(xhat_mults)):
            slopes[j] += cur_slope * xhat_mults[j]

        orthogonal_bases.append(orthogonal_base)

    return slopes[1:], slopes[0]


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

        self.cid_press   = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_motion  = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_key     = self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        self.update_fit()
        self.ax.set_title("Drag points. Press 'a' to add a point at the cursor.")
        self.fig.canvas.draw_idle()

    def update_fit(self):
        if len(self.x) >= 2:
            slopes, b = poly_regress(self.x, self.y, 3)

            xmin, xmax = min(self.x), max(self.x)

            if xmin == xmax:
                xmin -= 1.0
                xmax += 1.0
            xs = np.linspace(xmin, xmax, 200)

            ys = np.full_like(xs, b, dtype=float)
            for i, s in enumerate(slopes):
                ys += s * (xs ** (i + 1))

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


if __name__ == "__main__":
    fig, ax = plt.subplots()
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.grid(True, alpha=0.3)

    x0 = [-2, -1, 0, 1, 2]
    y0 = [-4, -2, 0, 2, 4]

    editor = PointEditor(ax, x0, y0)
    plt.show()
