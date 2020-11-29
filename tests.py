"""Module with testing in different datasets"""

from modules.data.datasets import (FibonacciDataset,
                                   FactorialDataset,
                                   PeriodDataset,
                                   ExponentialDataset)
from modules.network.jordan_network import Jordan
from modules.help.plots import draw_error_plot

from modules.train import train_model
from modules.evaluate import eval_model


def train_eval_fibonacci(verbose: bool = False) -> float:
    """
    Trains jordan model on fibonacci sequence and evaluates it.
    Sequence example: [1, 1, 2, 3, 5, 8, 13, 21, ...]

    :param verbose: if verbose draws plots
    :return: accuracy og the model
    """

    config = {
        'lr': 0.000003,
        'momentum': 0.1,
        'n_epochs': 500
    }

    dataset = FibonacciDataset(number_of_precalculated_values=12, number_of_input_elements=2)

    in_features = dataset.number_of_input_elements
    out_features = 1

    network = Jordan(lr=config['lr'],
                     momentum=config['momentum'],
                     shape=[in_features, 10, out_features])

    errors_list = train_model(network=network,
                              dataset=dataset,
                              n_epochs=config['n_epochs'])
    accuracy = eval_model(network=network,
                          dataset=dataset)

    if verbose:
        draw_error_plot(errors_list=errors_list, title='Jordan model on fibonacci sequence errors')

    return accuracy


def train_eval_period(verbose: bool = False) -> float:
    """
    Trains jordan model on periodical sequence and evaluates it.
    Sequence example: [1, 0, -1, 0, 1, 0, -1, ...]

    :param verbose: if verbose draws plots
    :return: accuracy og the model
    """

    config = {
        'lr': 0.003,
        'momentum': 0.1,
        'n_epochs': 500
    }

    dataset = PeriodDataset(number_of_precalculated_values=12, number_of_input_elements=3)

    in_features = dataset.number_of_input_elements
    out_features = 1

    network = Jordan(lr=config['lr'],
                     momentum=config['momentum'],
                     shape=[in_features, 25, 8, out_features])

    errors_list = train_model(network=network,
                              dataset=dataset,
                              n_epochs=config['n_epochs'])
    accuracy = eval_model(network=network,
                          dataset=dataset)

    if verbose:
        draw_error_plot(errors_list=errors_list, title='Jordan model on periodical sequence errors')

    return accuracy


def train_eval_factorial(verbose: bool = False) -> float:
    """
    Trains jordan model on factorial sequence and evaluates it.
    Sequence example: [1, 1, 2, 6, 24, 120, 720, ...]

    :param verbose: if verbose draws plots
    :return: accuracy og the model
    """

    config = {
        'lr': 0.000003,
        'momentum': 0.1,
        'n_epochs': 50_000
    }

    dataset = FactorialDataset(number_of_precalculated_values=5, number_of_input_elements=1)

    in_features = dataset.number_of_input_elements
    out_features = 1

    network = Jordan(lr=config['lr'],
                     momentum=config['momentum'],
                     shape=[in_features, 10, out_features])

    errors_list = train_model(network=network,
                              dataset=dataset,
                              n_epochs=config['n_epochs'])
    accuracy = eval_model(network=network,
                          dataset=dataset)

    if verbose:
        draw_error_plot(errors_list=errors_list, title='Jordan model on factorial sequence errors')

    return accuracy


def train_eval_exponent(verbose: bool = False) -> float:
    """
    Trains jordan model on exponential sequence and evaluates it.
    Sequence example: [0, 1, 4, 3, 16, 5, 36, 7, 64, ...]

    :param verbose: if verbose draws plots
    :return: accuracy of the model
    """

    config = {
        'lr': 0.00003,
        'momentum': 0.1,
        'n_epochs': 10_000
    }

    dataset = ExponentialDataset(number_of_precalculated_values=6, number_of_input_elements=3)

    in_features = dataset.number_of_input_elements
    out_features = 1

    network = Jordan(lr=config['lr'],
                     momentum=config['momentum'],
                     shape=[in_features, 10, out_features])

    errors_list = train_model(network=network,
                              dataset=dataset,
                              n_epochs=config['n_epochs'])
    accuracy = eval_model(network=network,
                          dataset=dataset)

    if verbose:
        draw_error_plot(errors_list=errors_list, title='Jordan model on exponential sequence errors')

    return accuracy


if __name__ == '__main__':
    verbose = True

    # accuracy_fibonacci = train_eval_fibonacci(verbose=verbose)
    # print(f'Fibonacci accuracy: {accuracy_fibonacci}')
    #
    # accuracy_period = train_eval_period(verbose=verbose)
    # print(f'Period accuracy: {accuracy_period}')

    accuracy_factorial = train_eval_factorial(verbose=verbose)
    print(f'Factorial accuracy: {accuracy_factorial}')

    # accuracy_exponent = train_eval_exponent(verbose=verbose)
    # print(f'Exponent accuracy: {accuracy_exponent}')
