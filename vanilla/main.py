import csv

import numpy as np

from accuracy import get_accuracy
from constructor import initiate_layer, full_forward_propagation, full_backward_propagation
from loss import cross_entropy
from optimizer import gradient_descent


RESCALE = 255

architecture = [
    {
        'input_dim': [784, ],
        'output_dim': [],
        'activation': 'relu',
    },
    {
        'input_dim': [],
        'output_dim': [],
        'activation': 'relu',
    },
    {
        'input_dim': [],
        'output_dim': [],
        'activation': 'sigmoid',
    },
]


def train(X, Y, nn_architecture, epochs, learning_rate):
    params_values = initiate_layer(nn_architecture, 2)
    cost_history = []
    accuracy_history = []

    for i in range(epochs):
        Y_hat, cashe = full_forward_propagation(X, params_values, nn_architecture)
        cost = cross_entropy(Y_hat, Y)
        cost_history.append(cost)
        accuracy = get_accuracy(Y_hat, Y)
        accuracy_history.append(accuracy)

        grads_values = full_backward_propagation(Y_hat, Y, cashe, params_values, nn_architecture)
        params_values = gradient_descent(params_values, grads_values, nn_architecture, learning_rate)

    return params_values, cost_history, accuracy_history


if __name__ == '__main__':
    data = np.genfromtxt('../data/train.csv', delimiter=',', skip_header=1)
    np.random.shuffle(data)

    m, n = data.shape

    X_train, Y_train = data[:, 1:], data[:, 0]

    # TODO: Cross-validation 구현
    # TODO: mini-batch 구현
    X = X.T / 255

    pass
