import numpy as np


def relu(Z):
    return np.maximum(0, Z)


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def d_relu(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


def d_sigmoid(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)
