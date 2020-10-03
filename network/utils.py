"""Module with help functions"""

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation function

    :param x: input matrix
    :return: resulted matrix
    """

    return 1 / (1 + np.exp(-x))


def sigmoid_der(x: np.ndarray) -> np.ndarray:
    """
    Derivative of sigmoid activation function

    :param x: input matrix
    :return: resulted matrix
    """

    return sigmoid(x) * (1 - sigmoid(x))


def log(x: np.ndarray) -> np.ndarray:
    """
    Log activation function (natural algorithm)

    :param x: input matrix
    :return: resulted matrix
    """

    return np.log(x)


def log_der(x: np.ndarray) -> np.ndarray:
    """
    Derivative of log activation function (natural algorithm)

    :param x: input matrix
    :return: resulted matrix
    """

    return 1 / x