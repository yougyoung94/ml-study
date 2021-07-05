import numpy as np

from activation import relu, sigmoid


def single_layer_forward_propagation(A_prev, W_current, b_current, activation="relu"):
    Z_current = np.dot(W_current, A_prev) + b_current

    if activation is "relu":
        activation_func = relu
    elif activation is "sigmoid":
        activation_func = activation
    else:
        raise Exception('Non-supported activation function')

    return activation_func(Z_current), Z_current
