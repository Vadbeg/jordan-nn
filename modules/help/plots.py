"""Module for drawing of the plots"""

from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def draw_error_plot(errors_list: List[float], title: str = 'Errors plot'):
    """
    Draws error plot of the network

    :param errors_list: listof errors
    :param title: title of the plot
    """

    plt.title(title)

    plt.xlabel('Epoch')
    plt.ylabel('Error')

    errors_list = errors_list[20:]

    plt.plot(list(range(len(errors_list))), errors_list)
    plt.show()


def draw_errors_for_all(errors_dict: Dict[str, List[float]]):
    """
    Draw errors for every list in dict on one plot

    :param errors_dict: dictionary with name of sequence and error list
    """

    fig, ax = plt.subplots(figsize=(12, 8))

    # for key in errors_dict.keys():
    #     errors_dict[key] = np.log10(errors_dict[key])

    plt.xlabel('Epoch')
    plt.ylabel('Error')

    # plt.yscale('log')

    plt.title(f'Epochs vs error for every sequence')

    sns.lineplot(data=errors_dict, ax=ax)

    plt.show()


