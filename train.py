import math

import numpy as np
from tqdm import tqdm

from network.jordan_network import Jordan
from data.datasets import FibonacciDataset
from data.base_dataset import BaseDataset


def train_model(network: Jordan, dataset: BaseDataset, n_epochs: int):
    tqdm_epochs = tqdm(range(n_epochs), postfix=f'Epochs...')

    for _ in tqdm_epochs:
        errors_epoch_list = list()

        for input_values, true_prediction in dataset:
            result = network.propagate_forward(x=input_values)

            error = network.propagate_backward(target=true_prediction)

            errors_epoch_list.append(error)

        tqdm_epochs.set_postfix(
            text=f'Epochs... Average error: {sum(errors_epoch_list) / len(errors_epoch_list):.2f}'
        )


def eval_mode(network: Jordan, dataset: BaseDataset):
    results_array = list()

    for input_values, true_prediction in dataset:
        result = network.propagate_forward(x=input_values)

        true_num = true_prediction.index(1)
        pred_num = result.argmax(axis=0)

        if true_num == pred_num:
            results_array.append(1)
        else:
            results_array.append(0)

    resulting_accuracy = sum(results_array) / len(results_array)

    return resulting_accuracy


if __name__ == '__main__':
    config = {
        'lr': 0.00001,
        'momentum': 0.1,
        'n_epochs': 200000
    }

    dataset = FibonacciDataset(max_dataset_length=7)

    in_features = dataset.max_value
    out_features = dataset.max_value

    print(in_features)

    network = Jordan(lr=config['lr'],
                     momentum=config['momentum'],
                     shape=[in_features, 15, out_features])

    train_model(network=network,
                dataset=dataset,
                n_epochs=config['n_epochs'])
    accuracy = eval_mode(network=network,
                         dataset=dataset)

    print(f'Accuracy: {accuracy}')