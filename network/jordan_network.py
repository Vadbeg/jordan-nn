"""Module with neural network code"""

from typing import List

import numpy as np


class Jordan:
    """Jordan network"""

    def __init__(self, *args):
        self.shape = args

        self.n_layers = len(args)

        self.layers = list()

        self.layers.append()

    def __init_weights__(self) -> List[np.ndarray]:
        """
        Neural networks he initialization

        :return: list of weights
        """

        layers = list()

        # it is jordan net, so + self.shape[-1]
        # first_layer = np.random.randn(self.shape[0] + self.shape[-1] + 1) + np.sqrt(2 / )
        # layers.append()


    def reset(self):
        pass

    def propagate_forward(self, data):
        pass

    def propagate_backward(self, target, lr, momentum):
        pass