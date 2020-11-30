"""Module with testing in different datasets"""

from typing import List, Tuple

from modules.data.datasets import (FibonacciDataset,
                                   FactorialDataset,
                                   PeriodDataset,
                                   ExponentialDataset)
from modules.network.jordan_network import Jordan
from modules.help.plots import draw_error_plot, draw_errors_for_all

from modules.train import train_model
from modules.evaluate import eval_model
from config import Config


def train_eval_fibonacci(verbose: bool = False) -> Tuple[float, List[float]]:
    """
    Trains jordan model on fibonacci sequence and evaluates it.
    Sequence example: [1, 1, 2, 3, 5, 8, 13, 21, ...]

    :param verbose: if verbose draws plots
    :return: accuracy og the model
    """

    dataset = FibonacciDataset(number_of_precalculated_values=Config.num_of_precalculated_values,
                               number_of_input_elements=Config.num_of_input_elements)

    in_features = dataset.number_of_input_elements
    out_features = 1

    network = Jordan(lr=Config.learning_rate,
                     momentum=Config.momentum,
                     shape=[in_features, Config.num_of_hidden_neurons, out_features])

    errors_list = train_model(network=network,
                              dataset=dataset,
                              n_epochs=Config.num_epochs)
    accuracy = eval_model(network=network,
                          dataset=dataset)

    if verbose:
        draw_error_plot(errors_list=errors_list, title='Jordan model on fibonacci sequence errors')

    return accuracy, errors_list


def train_eval_period(verbose: bool = False) -> Tuple[float, List[float]]:
    """
    Trains jordan model on periodical sequence and evaluates it.
    Sequence example: [1, 0, -1, 0, 1, 0, -1, ...]

    :param verbose: if verbose draws plots
    :return: accuracy og the model
    """

    dataset = PeriodDataset(number_of_precalculated_values=Config.num_of_precalculated_values,
                            number_of_input_elements=Config.num_of_input_elements)

    in_features = dataset.number_of_input_elements
    out_features = 1

    network = Jordan(lr=Config.learning_rate,
                     momentum=Config.momentum,
                     shape=[in_features, Config.num_of_hidden_neurons, out_features])

    errors_list = train_model(network=network,
                              dataset=dataset,
                              n_epochs=Config.num_epochs)
    accuracy = eval_model(network=network,
                          dataset=dataset)

    if verbose:
        draw_error_plot(errors_list=errors_list, title='Jordan model on periodical sequence errors')

    return accuracy, errors_list


def train_eval_factorial(verbose: bool = False) -> Tuple[float, List[float]]:
    """
    Trains jordan model on factorial sequence and evaluates it.
    Sequence example: [1, 1, 2, 6, 24, 120, 720, ...]

    :param verbose: if verbose draws plots
    :return: accuracy og the model
    """

    dataset = FactorialDataset(number_of_precalculated_values=Config.num_of_precalculated_values,
                               number_of_input_elements=Config.num_of_input_elements)

    in_features = dataset.number_of_input_elements
    out_features = 1

    network = Jordan(lr=Config.learning_rate,
                     momentum=Config.momentum,
                     shape=[in_features, Config.num_of_hidden_neurons, out_features])

    errors_list = train_model(network=network,
                              dataset=dataset,
                              n_epochs=Config.num_epochs)
    accuracy = eval_model(network=network,
                          dataset=dataset)

    if verbose:
        draw_error_plot(errors_list=errors_list, title='Jordan model on factorial sequence errors')

    return accuracy, errors_list


def train_eval_exponent(verbose: bool = False) -> Tuple[float, List[float]]:
    """
    Trains jordan model on exponential sequence and evaluates it.
    Sequence example: [0, 1, 4, 3, 16, 5, 36, 7, 64, ...]

    :param verbose: if verbose draws plots
    :return: accuracy of the model
    """

    dataset = ExponentialDataset(number_of_precalculated_values=Config.num_of_precalculated_values,
                                 number_of_input_elements=Config.num_of_input_elements)

    in_features = dataset.number_of_input_elements
    out_features = 1

    network = Jordan(lr=Config.learning_rate,
                     momentum=Config.momentum,
                     shape=[in_features, Config.num_of_hidden_neurons, out_features])

    errors_list = train_model(network=network,
                              dataset=dataset,
                              n_epochs=Config.num_epochs)
    accuracy = eval_model(network=network,
                          dataset=dataset)

    if verbose:
        draw_error_plot(errors_list=errors_list, title='Jordan model on exponential sequence errors')

    return accuracy, errors_list


if __name__ == '__main__':
    verbose = True

    accuracy_fibonacci, errors_list_fibonacci = train_eval_fibonacci(verbose=verbose)
    print(f'Fibonacci accuracy: {accuracy_fibonacci}')

    accuracy_period, errors_list_period = train_eval_period(verbose=verbose)
    print(f'Period accuracy: {accuracy_period}')

    accuracy_factorial, errors_list_factorial = train_eval_factorial(verbose=verbose)
    print(f'Factorial accuracy: {accuracy_factorial}')

    accuracy_exponent, errors_list_exponent = train_eval_exponent(verbose=verbose)
    print(f'Exponent accuracy: {accuracy_exponent}')

    # sequence_dict = {
    #     'fibonacci': errors_list_fibonacci[20:],
    #     'period': errors_list_period[20:],
    #     'factorial': errors_list_factorial[20:],
    #     'exponent': errors_list_exponent[20:]
    # }
    #
    # draw_errors_for_all(errors_dict=sequence_dict)
