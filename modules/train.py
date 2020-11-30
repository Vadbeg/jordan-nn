"""Model with functions for training"""

from typing import List

from tqdm import tqdm

from modules.network.jordan_network import Jordan
from modules.data.base_dataset import BaseDataset


def train_model(network: Jordan, dataset: BaseDataset, n_epochs: int) -> List[float]:
    """
    Performs model training

    :param network: network to train
    :param dataset: dataset for training
    :param n_epochs: number of  epochs we want to train
    :return: list of average epoch errors
    """

    tqdm_epochs = tqdm(range(n_epochs), postfix=f'Epochs...')

    total_error_list = list()

    for _ in tqdm_epochs:
        errors_epoch_list = list()

        for input_values, true_prediction in dataset:
            result = network.propagate_forward(x=input_values)

            error = network.propagate_backward(target=true_prediction)

            errors_epoch_list.append(error)

        average_error = sum(errors_epoch_list) / len(errors_epoch_list)

        tqdm_epochs.set_postfix(
            text=f'Epochs... Average error: {sum(errors_epoch_list) / len(errors_epoch_list):.2f}'
        )

        total_error_list.append(average_error)

    return total_error_list


def train_model_min_error(network: Jordan, dataset: BaseDataset,
                          n_epochs: int, min_error: float,
                          verbose: bool = True) -> List[float]:
    """
    Performs model training

    :param network: network to train
    :param dataset: dataset for training
    :param n_epochs: number of  epochs we want to train
    :param min_error: error we want to reach
    :param verbose: if True shows progress bar
    :return: list of average epoch errors
    """

    tqdm_epochs = tqdm(range(n_epochs), postfix=f'Epochs...', disable=not verbose)

    total_error_list = list()

    for _ in tqdm_epochs:
        errors_epoch_list = list()

        for input_values, true_prediction in dataset:
            result = network.propagate_forward(x=input_values)

            error = network.propagate_backward(target=true_prediction)

            errors_epoch_list.append(error)

        average_error = sum(errors_epoch_list) / len(errors_epoch_list)

        if average_error <= min_error:
            break

        tqdm_epochs.set_postfix(
            text=f'Epochs... Average error: {sum(errors_epoch_list) / len(errors_epoch_list):.2f}'
        )

        total_error_list.append(average_error)

    return total_error_list



