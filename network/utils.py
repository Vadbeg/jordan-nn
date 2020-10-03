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

    :url: http://jmlda.org/papers/doc/2011/no1/Rudoy2011Selection.pdf#page=12
    :param x: input matrix
    :return: resulted matrix
    """

    res = np.log(x + np.sqrt(x**2 + 1))
    res[x > 74.2] = 5
    res[x < -74.2] = -5
    res = res / 5

    return res


def log_der(x: np.ndarray) -> np.ndarray:
    """
    Derivative of log activation function (natural algorithm)

    :param x: input matrix
    :return: resulted matrix
    """

    res = 1 / (np.sqrt(x**2 + 1))
    res[x > 74.2] = 0
    res[x < -74.2] = 0
    res = res / 5

    return res
