"""Module with training"""

from typing import Dict

from modules.data.datasets import (FactorialDataset,
                                   FibonacciDataset,
                                   PeriodDataset,
                                   ExponentialDataset)
from modules.network.jordan_network import Jordan
from modules.train import train_model, train_model_min_error
from modules.evaluate import eval_model
from config import Config


if __name__ == '__main__':
    config = Config()

    datasets_mapping: Dict = {
        'factorial': FactorialDataset,
        'fibonacci': FibonacciDataset,
        'period': PeriodDataset,
        'exponent': ExponentialDataset
    }

    dataset_class = datasets_mapping[config.dataset]

    dataset = dataset_class(number_of_precalculated_values=config.num_of_precalculated_values,
                            number_of_input_elements=config.num_of_input_elements)

    in_features = dataset.number_of_input_elements
    out_features = 1

    network = Jordan(lr=config.learning_rate,
                     momentum=config.momentum,
                     shape=[in_features, config.num_of_hidden_neurons, out_features])

    # errors_list = train_model(network=network,
    #                           dataset=dataset,
    #                           n_epochs=config.num_epochs)
    errors_list = train_model_min_error(network=network,
                                        dataset=dataset,
                                        n_epochs=config.num_epochs,
                                        min_error=config.min_error)

    accuracy = eval_model(network=network,
                          dataset=dataset)

    print(f'Accuracy: {accuracy}')
