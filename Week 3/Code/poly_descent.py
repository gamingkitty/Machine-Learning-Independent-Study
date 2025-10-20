import random

degree = 4

slopes = [random.random() - 0.5 for i in range(degree + 1)]

def f(x_data):
    global degree
    return [sum([slopes[i] * x ** i for i in range(degree + 1)]) for x in x_data]

def loss(x_data, y_data):
    y_hat = f(x_data)
    return sum([((y_data[i] - y_hat[i]) ** 2) / len(x_data) for i in range(len(x_data))])

def get_gradient(x_data, y_data):
    y_hat = f(x_data)
    loss_deriv = [-2 * (y_data[i] - y_hat[i]) / len(x_data) for i in range(len(y_hat))]

    derivs = []
    for i in range(len(slopes)):
        deriv = sum([loss_deriv[j] * (x_data[j] ** i) for j in range(len(loss_deriv))])

        derivs.append(deriv)

    return derivs

def main():
    x_data = [1, 2, 3, 4, 5, 6]
    y_data = [2, 4, 6, 8, 10, 12]

    learning_rate = 0.000001

    for j in range(5000000):
        gradient = get_gradient(x_data, y_data)

        for i in range(len(slopes)):
            slopes[i] -= learning_rate * gradient[i]

        if j % 10000 == 0:
            print(f"Loss: {loss(x_data, y_data)}")

    print(slopes)


if __name__ == "__main__":
    main()