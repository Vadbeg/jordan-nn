"""Module with testing in different datasets"""

from data.datasets import (FibonacciDataset,
                           FactorialDataset,
                           PeriodDataset,
                           ExponentialDataset)
from network.jordan_network import Jordan
from help.plots import draw_error_plot

from train import train_model
from evaluate import eval_model


def train_eval_fibonacci(verbose: bool = False) -> float:
    """
    Trains jordan model on fibonacci sequence and evaluates it.
    Sequence example: [1, 1, 2, 3, 5, 8, 13, 21, ...]

    :param verbose: if verbose draws plots
    :return: accuracy og the model
    """

    config = {
        'lr': 0.003,
        'momentum': 0.1,
        'n_epochs': 500
    }

    dataset = FibonacciDataset(max_dataset_length=12)

    in_features = dataset.max_value * dataset.sequence_length
    out_features = dataset.max_value

    network = Jordan(lr=config['lr'],
                     momentum=config['momentum'],
                     shape=[in_features, 100, out_features])

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
        'n_epochs': 5000
    }

    dataset = PeriodDataset(max_dataset_length=39)

    in_features = len(dataset.ohe_mapped_values) * dataset.sequence_length
    out_features = len(dataset.ohe_mapped_values)

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
        'lr': 0.0003,
        'momentum': 0.1,
        'n_epochs': 5000
    }

    dataset = FactorialDataset(max_dataset_length=5)

    in_features = dataset.max_value
    out_features = dataset.max_value

    network = Jordan(lr=config['lr'],
                     momentum=config['momentum'],
                     shape=[in_features, 100, out_features])

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
        'lr': 0.0003,
        'momentum': 0.1,
        'n_epochs': 5000
    }

    dataset = ExponentialDataset(max_dataset_length=5)

    in_features = dataset.max_value
    out_features = dataset.max_value

    network = Jordan(lr=config['lr'],
                     momentum=config['momentum'],
                     shape=[in_features, 100, out_features])

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

    accuracy_fibonacci = train_eval_fibonacci(verbose=verbose)
    print(f'Fibonacci accuracy: {accuracy_fibonacci}')

    accuracy_period = train_eval_period(verbose=verbose)
    print(f'Period accuracy: {accuracy_period}')

    accuracy_factorial = train_eval_factorial(verbose=verbose)
    print(f'Factorial accuracy: {accuracy_factorial}')

    accuracy_exponent = train_eval_exponent(verbose=verbose)
    print(f'Exponent accuracy: {accuracy_exponent}')
