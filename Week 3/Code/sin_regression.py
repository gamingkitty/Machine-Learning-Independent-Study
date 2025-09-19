import math
import random
import matplotlib.pyplot as plt
import numpy as np

# Do gradient descent to find regression for Asin(Bx + C) + D

def gradient_descent(x_data, y_data):
    n = len(x_data)

    lowest_loss = 99999
    la, lb, lc, ld = 0, 0, 0, 0
    for _ in range(40):
        a = random.random() - 0.5
        b = random.random() - 0.5
        c = random.random() - 0.5
        d = (sum(y_data) / n) + random.random() - 0.5

        learning_rate = 0.004

        # C = sum((y - (Asin(Bx + C) + D))^2)

        final_loss = 0

        for _ in range(1000):
            sines = [math.sin((b * x) + c) for x in x_data]
            cosines = [a * math.cos((b * x) + c) for x in x_data]

            loss = sum([(y_data[i] - (a * sines[i] + d)) ** 2 for i in range(n)])
            final_loss = loss

            losses = [(y_data[i] - (a * sines[i] + d)) for i in range(n)]

            dc_da = -2 * sum([losses[i] * sines[i] for i in range(n)])
            dc_db = -2 * sum([losses[i] * cosines[i] * x_data[i] for i in range(n)])
            dc_dc = -2 * sum([losses[i] * cosines[i] for i in range(n)])
            dc_dd = -2 * sum([losses[i] for i in range(n)])

            a -= learning_rate * dc_da
            b -= learning_rate * dc_db
            c -= learning_rate * dc_dc
            d -= learning_rate * dc_dd

        if final_loss < lowest_loss:
            la, lb, lc, ld = a, b, c, d
            lowest_loss = final_loss

    return la, lb, lc, ld


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
        self.ax.set_title("Press 'a' to add a point at the cursor.")
        self.fig.canvas.draw_idle()

    def update_fit(self):
        if len(self.x) >= 2:
            a, b, c, d = gradient_descent(self.x, self.y)
            xmin, xmax = min(self.x), max(self.x)

            if xmin == xmax:
                xmin -= 1.0
                xmax += 1.0
            xs = np.linspace(xmin, xmax, 200)

            ys = np.full_like(xs, 0, dtype=float)
            ys += [a * math.sin((b * x) + c) + d for x in xs]

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
    ax.set_xlim(-1, 1)
    ax.set_ylim(-3, 3)
    ax.grid(True, alpha=0.3)

    x_data = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    y_data = [0, 1, 2, 1, 0, -1, -2, -1, 0]

    editor = PointEditor(ax, x_data, y_data)
    plt.show()


