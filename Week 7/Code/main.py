import numpy as np
import math
import dense_layer
import operations as ops
from keras.datasets import mnist


def relu(x):
    return [max(0, e) for e in x]


def relu_derivative(x):
    return [e > 0 for e in x]


def softmax(x):
    s = sum(x)
    return [math.exp(e) / s for e in x]


def softmax_derivative(z):
    s = softmax(z)
    return [[s[i] * (int(i == j) - s[j]) for j in range(len(s))] for i in range(len(s))]


def loss(label, out):
    return sum((label[i] - out[i]) ** 2 for i in range(len(label)))


def loss_derivative(label, out):
    return ops.mult_scalar(ops.sub(out, label), 2)


def main():
    input_size = 784
    output_size = 10
    layers = [dense_layer.Layer(input_size, 16, relu, relu_derivative, True),
              dense_layer.Layer(16, output_size, softmax, softmax_derivative, False)]

    epochs = 10
    learning_rate = 0.01

    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_labels = np.eye(10)[train_labels]
    train_images = train_images.reshape(-1, 784)

    training_data = train_images / 255.0
    training_labels = train_labels

    print("Training...")
    for i in range(epochs):
        total_loss = 0
        for j in range(len(training_data)):
            out = training_data[j]
            label = training_labels[j]
            a_data = [out]
            z_data = []
            for layer in layers:
                z, out = layer.forward_pass(out)
                z_data.append(z)
                a_data.append(out)
            total_loss += loss(label, out)

            dc_da = loss_derivative(label, out)


            for k in reversed(range(len(layers))):
                layer = layers[k]
                dc_da = layer.backwards_pass(a_data[k], z_data[k], dc_da)

            for layer in layers:
                layer.apply_gradient(learning_rate)

            if j % 1 == 0:
                print("epoch:", j, "loss:", total_loss / (j + 1))



if __name__ == "__main__":
    main()
