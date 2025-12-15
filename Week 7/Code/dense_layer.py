import math
import random
import operations as ops


class Layer:
    def __init__(self, input_num, neuron_num, activation_function, activation_derivative, is_elementwise):
        self.input_num = input_num
        self.neuron_num = neuron_num
        self.weights = [[(random.random() - 0.5) * math.sqrt(2 / input_num) for _ in range(input_num)] for _ in range(neuron_num)]
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative
        self.is_elementwise = is_elementwise

        self.gradient = [ops.zeros(input_num) for _ in range(neuron_num)]

    # Returns z_data (output before activation) and a_data (output after activation)
    def forward_pass(self, input):
        # Compute weights * input
        z_data = [ops.dot(input, neuron_weights) for neuron_weights in self.weights]
        # Apply activation
        return z_data, self.activation_function(z_data)

    # If not elementwise format is:
    # [[z_1 -> a_1, z_1 -> a_2]
    #  [z_2 -> a_1, z_2 -> a_2]]
    def backwards_pass(self, inputs, z_data, d_cost_d_out):
        d_a_d_z = self.activation_derivative(z_data)
        if self.is_elementwise:
            # d_cost_d_weights = [ops.mult_scalar(input, d_cost_d_out[i] * d_a_d_z[i]) for i in range(len(d_cost_d_out))]
            d_cost_d_z = ops.mult(d_a_d_z, d_cost_d_out)
            d_cost_d_weights = ops.matrix_mult(ops.transpose(d_cost_d_z), inputs)

            d_cost_d_in = ops.matrix_mult(d_cost_d_z, self.weights)[0]
            # d_out_d_in = [ops.mult_scalar(self.weights[i], d_cost_d_out[i] * d_a_d_z[i]) for i in range(self.neuron_num)]
            # # Sum along columns
            # d_cost_d_in = [sum(col) for col in zip(*d_out_d_in)]
        else:
            # print(len(d_a_d_z))
            # print(len(inputs))
            # Something wrong here, shapes don't match
            # d_cost_d_weights = [ops.mult_scalar(ops.mult(inputs, d_a_d_z[i]), d_cost_d_out[i]) for i in range(len(d_cost_d_out))]
            d_cost_d_z = ops.matrix_mult(d_cost_d_out, d_a_d_z)
            d_cost_d_weights = ops.matrix_mult(ops.transpose(d_cost_d_z), inputs)

            # d_z_d_cost = [ops.dot(d_a_d_z[i], d_cost_d_out) for i in range(len(d_a_d_z))]
            #
            # d_cost_d_in = [ops.mult_scalar(self.weights[i], d_z_d_cost[i]) for i in range(self.neuron_num)]
            # # Sum along columns
            # d_cost_d_in = [sum(col) for col in zip(*d_cost_d_in)]
            d_cost_d_in = ops.matrix_mult(d_cost_d_z, self.weights)[0]

        # print(len(d_cost_d_weights[0]))
        self.gradient = d_cost_d_weights

        return d_cost_d_in

    def apply_gradient(self, learning_rate):
        self.weights = [ops.sub(self.weights[i], ops.mult_scalar(self.gradient[i], learning_rate)) for i in range(self.neuron_num)]


