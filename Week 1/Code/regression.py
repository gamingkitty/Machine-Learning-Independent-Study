import math

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
    def __init__(
        self,
        ax,
        x_init=None,
        y_init=None,
        *,
        enable_drag=True,
        title="Data & Fit",
        xlabel="x",
        ylabel="y",
        coeff_precision=3,
        r2_precision=4
    ):
        self.ax = ax
        self.fig = ax.figure

        self.enable_drag = bool(enable_drag)
        self.coeff_precision = int(coeff_precision)
        self.r2_precision = int(r2_precision)

        self.function_names = function_names  # may be None; handled during formatting

        self.x = list(x_init if x_init is not None else [0, 1, 2])
        self.y = list(y_init if y_init is not None else [0, 1, 4])

        # scatter points
        self.points = []
        for xi, yi in zip(self.x, self.y):
            p, = ax.plot([xi], [yi], 'o', ms=9, picker=5)
            self.points.append(p)

        # fitted curve
        self.fit_line, = ax.plot([], [], '-', lw=2, label="fit")

        # info text (equation + R^2)
        self.info_text = ax.text(
            0.02, 0.98, "", transform=ax.transAxes, va='top', ha='left'
        )

        # titles
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)

        # drag state
        self.drag_i = None
        self.drag_anchor = None

        # events
        self.cid_press  = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cid_release= self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_key    = self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        # initial fit
        self.update_fit()
        self.fig.canvas.draw_idle()

    # ----- public knobs -----
    def set_drag(self, enabled: bool):
        self.enable_drag = bool(enabled)

    def set_titles(self, *, title=None, xlabel=None, ylabel=None):
        if title is not None:
            self.ax.set_title(title)
        if xlabel is not None:
            self.ax.set_xlabel(xlabel)
        if ylabel is not None:
            self.ax.set_ylabel(ylabel)
        self.redraw()

    def set_function_names(self, names):
        self.function_names = list(names) if names is not None else None
        self.update_fit()
        self.redraw()

    def update_fit(self):
        if len(self.x) >= 2:
            X = np.vstack([f(np.asarray(self.x, dtype=float)) for f in functions]).T
            y = np.asarray(self.y, dtype=float)

            coeffs = function_regress([f(np.asarray(self.x, dtype=float)) for f in functions], y)
            coeffs = np.asarray(coeffs, dtype=float)

            print(coeffs)

            # Smooth line
            xmin, xmax = float(min(self.x)), float(max(self.x))
            if xmin == xmax:
                xmin -= 1.0
                xmax += 1.0
            xs = np.linspace(xmin, xmax, 200)
            ys = np.zeros_like(xs, dtype=float)
            for i, f in enumerate(functions):
                ys += coeffs[i] * f(xs)

            self.fit_line.set_data(xs, ys)

            # R^2 on the *data*:
            y_hat = X.dot(coeffs)
            ss_res = float(np.sum((y - y_hat) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

            # Update equation + R^2 label
            eq = self._format_equation(coeffs)
            info = f"{eq}\nR² = {r2:.{self.r2_precision}f}"
            self.info_text.set_text(info)
        else:
            self.fit_line.set_data([], [])
            self.info_text.set_text("Add at least two points to fit.")

    def _format_equation(self, coeffs):
        # Build "y = a·x^2 + b·x + c" using function_names when provided.
        # Empty name "" denotes intercept term.
        prec = self.coeff_precision
        names = self.function_names
        if names is None or len(names) != len(coeffs):
            # Best-effort auto-names: f0, f1, ..., with last as intercept if constant function detected
            names = [f"f{i}" for i in range(len(coeffs))]

        terms = []
        for c, nm in zip(coeffs, names):
            c_str = f"{c:.{prec}f}"
            if nm == "" or nm is None:
                term = c_str
            else:
                # coefficient * basis name; handle ± in join later
                # Use "·" for multiplication only when |c| not ~1 or name looks like constant-free
                if np.isclose(abs(float(c_str)), 1.0, atol=10**(-prec)):
                    # show just sign if magnitude ~1 (e.g., "-x", "+x^2")
                    sign = "-" if float(c_str) < 0 else "+"
                    # remove "1"
                    term = f"{sign} {nm}"
                else:
                    sign = "-" if float(c_str) < 0 else "+"
                    term = f"{sign} {abs(float(c_str)):.{prec}f}·{nm}"
            terms.append(term)

        if not terms:
            return "y = 0"
        first = terms[0]
        if first.startswith("+ "):
            terms[0] = first[2:]
        return "y = " + " ".join(terms)

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
        if not self.enable_drag:
            return
        if event.inaxes != self.ax or event.button != 1:
            return
        i = self.pick_point_index(event)
        if i is None:
            return
        x0, y0 = self.x[i], self.y[i]
        self.drag_i = i
        self.drag_anchor = (x0, y0, event.xdata, event.ydata)

    def on_motion(self, event):
        if not self.enable_drag:
            return
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
        if not self.enable_drag:
            return
        if self.drag_i is None:
            return
        self.drag_i = None
        self.drag_anchor = None
        self.update_fit()
        self.redraw()

    def on_key(self, event):
        # keep 'a' to add point at cursor
        if event.key == 'a' and event.inaxes == self.ax and event.xdata is not None:
            self.add_point(event.xdata, event.ydata)


def intercept(x_data):
    return ops.ones(len(x_data))


def linear(x_data):
    return x_data


def sqrt(x_data):
    return ops.power(x_data, 1/2)


def sqr(x_data):
    return ops.power(x_data, 2)


def cbrt(x_data):
    return ops.power(x_data, 1/3)


def cube(x_data):
    return ops.power(x_data, 3)

def sin(x_data):
    return [math.sin(x) for x in x_data]


functions = [linear, sin, cube, cbrt, intercept]
function_names = ["x", "sin", "x^3", "x^(1/3)", ""]


if __name__ == "__main__":
    fig, ax = plt.subplots()
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    x0 = [0.08, 0.16, 0.24, 0.32, 0.4]
    y0 = [0.17, 0.34, 0.52, 0.70, 0.86]

    editor = PointEditor(ax, x0, y0)
    plt.show()

    print()


