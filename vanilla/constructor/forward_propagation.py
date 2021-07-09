import numpy as np

from activation import relu, sigmoid


def single_layer_forward_propagation(A_prev, W_current, b_current, activation='relu'):
    Z_current = np.dot(W_current, A_prev) + b_current

    if activation == 'relu':
        activation_func = relu
    elif activation == 'sigmoid':
        activation_func = sigmoid
    else:
        raise Exception('Non-supported activation function')

    return activation_func(Z_current), Z_current


def full_forward_propagation(X, param_values, architecture):
    memory = {}
    A_curr = X

    for idx, layer in enumerate(architecture):
        layer_idx = idx + 1
        A_prev = A_curr

        activ_function_curr = layer['activation']
        W_curr = param_values['W' + str(layer_idx)]
        b_curr = param_values['b' + str(layer_idx)]
        A_curr, Z_curr = single_layer_forward_propagation(A_prev, W_curr, b_curr, activ_function_curr)

        memory['A' + str(idx)] = A_prev
        memory['Z' + str(layer_idx)] = Z_curr

    return A_curr, memory
