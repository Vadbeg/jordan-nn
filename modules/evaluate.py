"""Module with evaluation of NN"""

from modules.network.jordan_network import Jordan
from modules.data.base_dataset import BaseDataset


def eval_model(network: Jordan, dataset: BaseDataset):
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
