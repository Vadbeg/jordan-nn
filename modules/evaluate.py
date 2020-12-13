"""Module with evaluation of NN"""

import math

from modules.network.jordan_network import Jordan
from modules.data.base_dataset import BaseDataset


def eval_model(network: Jordan, dataset: BaseDataset):
    results_array = list()

    for input_values, true_prediction in dataset:
        result = network.propagate_forward(x=input_values)
        result = result.tolist()

        result = [int(round(curr_result)) for curr_result in result]

        try:
            print(f'Input: {input_values}, Pred num: {result}. True: {true_prediction}')

            if true_prediction == result:
                results_array.append(1)
            else:
                results_array.append(0)
        except ValueError as err:
            results_array.append(0)

    resulting_accuracy = sum(results_array) / len(results_array)

    return resulting_accuracy
