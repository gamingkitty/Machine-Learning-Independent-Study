import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def main():
    # coefficients of the different powers of x; also stores the degree of the polynomial
    # for example, coeffs = [2, 0, 3] represents the equation 2 + 0*x + 3*x^2
    coeffs = [1, 1]
    # x-values we will use to train the model
    x = [random.randint(-10, 10) for _ in range(100)]
    # y-values obtained by putting the x-value through the equation
    y = [sum(c * v**i for i, c in enumerate(coeffs)) for v in x]
    # actually one more than the degree, but this makes it easier to use in loops
    degree = len(coeffs)

    # randomly initialize the weights
    weights = [random.random() - 0.5 for _ in range(degree)]
    learning_rate = 0.00003
    epochs = 5000  # total number of gradient steps to perform (like: for _ in range(epochs):)

    # Visualization setup using matplotlib (true function in green, model in purple)
    x_min, x_max = min(x) - 1, max(x) + 1
    x_plot = np.linspace(x_min, x_max, 400)
    y_true_plot = np.zeros_like(x_plot)
    for i, c in enumerate(coeffs):
        y_true_plot += c * (x_plot ** i)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot the true function (green)
    true_line, = ax.plot(x_plot, y_true_plot, color='green', linewidth=2.0, label='True function')

    # Initial predicted curve (purple)
    y_pred_plot = np.zeros_like(x_plot)
    for i in range(degree):
        y_pred_plot += weights[i] * (x_plot ** i)
    pred_line, = ax.plot(x_plot, y_pred_plot, color='purple', linewidth=2.0, label='Model (gradient descent)')

    # Scatter training points for context (light gray)
    ax.scatter(x, y, s=16, color='gray', alpha=0.35, label='Training points')
    ax.legend(loc='best')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Polynomial Regression via Gradient Descent')

    # Precompute powers for training x for efficiency: init_vals[j][i] = x_j^i
    # find x^i for each i from 0 to the degree of the polynomial, for each x-value
    init_vals = [[v ** i for i in range(degree)] for v in x]

    # Helper: compute prediction for training data given current weights
    def predict_train(cur_weights):
        # plug the init_vals into polynomials with the current weights as coefficients to get predictions
        return [sum(cur_weights[i] * init_vals[j][i] for i in range(degree)) for j in range(len(x))]

    # Helper to compute loss (mean squared error)
    def mse(pred):
        # mean squared error
        return sum((y[i] - pred[i]) ** 2 for i in range(len(y))) / len(y)

    # Live text for the current loss to make progress visible
    loss_val = mse(predict_train(weights))
    loss_text = ax.text(0.02, 0.98, f'Loss: {loss_val:.6f}', transform=ax.transAxes,
                        va='top', ha='left', fontsize=10, color='black')

    # Make sure y-limits comfortably contain both true and initial predicted curves
    y_all = np.concatenate([y_true_plot, y_pred_plot, np.asarray(y)])
    y_pad = max(1.0, 0.05 * (y_all.max() - y_all.min() + 1e-9))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_all.min() - y_pad, y_all.max() + y_pad)

    steps_per_frame = 10   # do several gradient steps per frame so it moves smoothly but not too slow
    max_frames = (epochs + steps_per_frame - 1) // steps_per_frame  # derive total frames from epochs
    tol = 1e-10            # early stop threshold on loss improvement

    # We will mutate weights inside the update function
    nonlocal_state = {'w': weights, 'step': 0}

    def update(_frame_idx):
        # chain rule in gradient descent; dc_da * da_dw = dc_dw.
        # dc_da is just -2 * (y - pred) for each point, because that point contributes (y - pred)^2 to the loss
        # da_dw is just x^i for each point, which we already computed in init
        # this is because a is computed as w_i * x^i
        w = nonlocal_state['w']
        remaining = epochs - nonlocal_state['step']
        if remaining <= 0:
            # Update plot with current weights without further training
            y_pred_plot = np.zeros_like(x_plot)
            for i in range(degree):
                y_pred_plot += w[i] * (x_plot ** i)
            pred_line.set_ydata(y_pred_plot)
            cur_loss = mse(predict_train(w))
            loss_text.set_text(f'Loss: {cur_loss:.6f} (done)')
            return pred_line, loss_text

        steps_this_frame = min(steps_per_frame, remaining)
        prev_loss = None
        for _ in range(steps_this_frame):
            pred_train = predict_train(w)
            loss = mse(pred_train)
            prev_loss = loss if (prev_loss is None) else prev_loss
            dc_da = [-2 * (y[j] - pred_train[j]) for j in range(len(pred_train))]
            gradients = [sum(dc_da[j] * init_vals[j][i] for j in range(len(x))) / len(x) for i in range(degree)]
            w = [w[i] - gradients[i] * learning_rate for i in range(len(w))]
        nonlocal_state['w'] = w
        nonlocal_state['step'] += steps_this_frame

        # Update the predicted curve for plotting
        y_pred_plot = np.zeros_like(x_plot)
        for i in range(degree):
            y_pred_plot += w[i] * (x_plot ** i)
        pred_line.set_ydata(y_pred_plot)

        # Update loss display (current loss after the chunk of steps)
        cur_loss = mse(predict_train(w))
        loss_text.set_text(f'Loss: {cur_loss:.6f}  Step: {nonlocal_state['step']}/{epochs}')

        return pred_line, loss_text

    anim = FuncAnimation(
        fig,
        update,
        frames=max_frames,
        interval=30,  # milliseconds between frames; small for smooth updates
        blit=True,
        repeat=False,
    )

    # Show the animation
    plt.tight_layout()
    plt.show()

    # After animation, print final weights to console for reference
    # mean squared error is minimized when these match coeffs closely
    print("Final weights:", nonlocal_state['w'])


if __name__ == "__main__":
    main()