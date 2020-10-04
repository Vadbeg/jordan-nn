"""Module with help functions"""

from typing import Union, List

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation function

    :param x: input matrix
    :return: resulted matrix
    """

    return 1 / (1 + np.exp(-x))


def softmax(x: Union[np.ndarray, List]) -> np.ndarray:
    """
    Softmax activation function

    :param x: input matrix
    :return: resulted matrix
    """
    e_x = np.exp(x - np.max(x))

    return e_x / e_x.sum(axis=0)


def softmax_der(x: Union[np.ndarray, List]) -> np.ndarray:
    """
    Derivative of softmax activation function

    :param x: input matrix
    :return: resulted matrix
    """
    res = softmax(x) * (1 - softmax(x))

    return res


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


def linear(x: np.ndarray) -> np.ndarray:
    """
    Linear activation function (natural algorithm)

    :url: http://jmlda.org/papers/doc/2011/no1/Rudoy2011Selection.pdf#page=12
    :param x: input matrix
    :return: resulted matrix
    """

    return x


def linear_der(x: np.ndarray) -> Union[np.ndarray, int]:
    """
    Derivative of log activation function (natural algorithm)

    :param x: input matrix
    :return: resulted matrix
    """

    return 1


def cross_entropy_loss(y_pred: Union[np.ndarray, List],
                       y_true: Union[np.ndarray, List]) -> List[Union[float, np.ndarray]]:
    """
    Cross entropy loss.
    Use it after sigmoid function!

    :param y_pred: values predicted by NN (with softmax on top of it)
    :param y_true: true value
    :return: loss value
    """
    y_pred = np.array(y_pred, dtype=np.float16)
    # y_pred = softmax(x=y_pred)

    y_true = np.array(y_true, dtype=np.float16)
    y_true_argmax = y_true.argmax(axis=0)

    y_pred[y_true_argmax] = np.clip(y_pred[y_true_argmax], a_min=0.0001, a_max=None)

    log_likelihood = - np.log(y_pred[y_true_argmax])
    loss = np.sum(log_likelihood)

    return loss


def cross_entropy_loss_der(y_pred: Union[np.ndarray, List],
                           y_true: Union[np.ndarray, List]) -> Union[List[float], np.ndarray]:
    """
    Cross entropy loss derivative.
    Use it after sigmoid function!

    :param y_pred: values predicted by NN (with softmax, sigmoid on top of it)
    :param y_true: true value
    :return: loss value
    """
    y_pred = np.array(y_pred, dtype=np.float16)
    grad = linear(x=y_pred)  # TODO: rebuild it !!!

    y_true = np.array(y_true, dtype=np.float16)
    y_true_argmax = y_true.argmax(axis=0)

    grad[y_true_argmax] = np.clip(grad[y_true_argmax], a_min=0.0001, a_max=None)
    grad[y_true_argmax] -= 1

    step = - grad

    return step


if __name__ == '__main__':
    """
[ 35.29  78.37 -24.12]
[0 0 1]
Y_pred: [0. 1. 0.]
Last layer der: [-0. -0.  0.]
    """

    y_pred = [0, 0, 0, -10000]
    y_true = [0, 0, 0, 1]

    y_pred = [35.29, 78.37, -24.12]
    y_true = [0, 0, 1]

    # res = cross_entropy_loss(y_pred=y_pred, y_true=y_true)
    # print(f'Loss value: {res}')

    res_der = cross_entropy_loss_der(y_pred=y_pred, y_true=y_true)
    print(f'Derivative value: {res_der.round(2)}')
