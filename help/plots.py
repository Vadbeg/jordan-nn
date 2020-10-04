"""Module for drawing of the plots"""

from typing import List

import matplotlib.pyplot as plt


def draw_error_plot(errors_list: List[float], title: str = 'Errors plot'):
    """
    Draws error plot of the network

    :param errors_list: listof errors
    :param title: title of the plot
    """

    plt.title(title)

    plt.plot(list(range(len(errors_list))), errors_list)
    plt.show()
