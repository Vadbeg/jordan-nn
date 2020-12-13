"""Module for plots building"""

from typing import Dict

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from modules.network.jordan_network import Jordan
from modules.data.datasets import CustomDataset
from modules.data.utils import (generate_exp_values, generate_fibonacci_values,
                                generate_period_values, generate_factorial_values)
from modules.train import train_model_min_error
from config import Config


def perform_pipeline_for_plots():
    config = Config()

    data_mapping: Dict = {
        'factorial': generate_factorial_values(number_of_precalculated_values=config.num_of_precalculated_values),
        'fibonacci': generate_fibonacci_values(number_of_precalculated_values=config.num_of_precalculated_values),
        'period': generate_period_values(number_of_precalculated_values=config.num_of_precalculated_values),
        'exponent': generate_exp_values(number_of_precalculated_values=config.num_of_precalculated_values),
        'custom': config.data
    }

    dataset = CustomDataset(data=data_mapping[config.data_type],
                            number_of_input_elements=config.num_of_input_elements,
                            number_of_output_elements=config.num_of_output_elements)

    in_features = dataset.number_of_input_elements
    out_features = dataset.number_of_output_elements

    network = Jordan(lr=config.learning_rate,
                     momentum=config.momentum,
                     make_zero_context=config.make_zero_context,
                     shape=[in_features, config.num_of_hidden_neurons, out_features])

    errors_list = train_model_min_error(network=network,
                                        dataset=dataset,
                                        n_epochs=Config.num_epochs,
                                        min_error=Config.min_error,
                                        verbose=True)

    return errors_list


def learning_rate_epochs_plot():
    learning_rate_list = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01]

    num_of_epochs_list = list()

    for curr_learning_rate in tqdm(learning_rate_list, postfix=f'Training networks'):
        Config.learning_rate = curr_learning_rate

        total_error_list = perform_pipeline_for_plots()

        num_of_epochs_list.append(len(total_error_list))

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.set_title(f'Learning rate vs number of epochs to achieve {Config.min_error} MSE error')

    ax.set_xlabel('Learning rate')
    ax.set_ylabel('Number of epochs')

    sns.lineplot(x=learning_rate_list, y=num_of_epochs_list, ax=ax)

    plt.show()


def errors_epochs_plot():
    errors_list = [0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.1, 0.2, 0.4, 0.8, 1.0]

    num_of_epochs_list = list()

    for curr_error in tqdm(errors_list, postfix=f'Training networks'):
        Config.min_error = curr_error

        total_error_list = perform_pipeline_for_plots()

        num_of_epochs_list.append(len(total_error_list))

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.set_title(f'Error vs number of epochs to achieve this MSE error')

    ax.set_xlabel('Error')
    ax.set_ylabel('Number of epochs')

    sns.lineplot(x=errors_list, y=num_of_epochs_list, ax=ax)

    plt.show()


def number_of_input_values_epochs_plot():
    num_of_input_values_list = [1, 2, 3, 4]

    num_of_epochs_list = list()

    for curr_num_of_input_values in tqdm(num_of_input_values_list, postfix=f'Training networks'):
        Config.num_of_input_elements = curr_num_of_input_values

        total_error_list = perform_pipeline_for_plots()

        num_of_epochs_list.append(len(total_error_list))

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.set_title(f'Num of input values vs number of epochs to achieve {Config.min_error} error')

    ax.set_xlabel('Number of input values')
    ax.set_ylabel('Number of epochs')

    sns.lineplot(x=num_of_input_values_list, y=num_of_epochs_list, ax=ax)

    plt.show()


def sequences_num_of_epochs_plot():
    datasets_list = ['factorial', 'fibonacci', 'period', 'exponent']

    min_errors_list = list()

    for curr_dataset in tqdm(datasets_list, postfix=f'Training networks'):
        Config.data_type = curr_dataset

        total_error_list = perform_pipeline_for_plots()

        min_errors_list.append(round(min(total_error_list), 2))

    print(min_errors_list)

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.set_title(f'Sequence vs min error achieved for {Config.num_epochs} epochs')

    ax.set_xlabel('Sequence')
    ax.set_ylabel('Min error')

    barplot_sequences = sns.barplot(x=datasets_list, y=min_errors_list, ax=ax)

    for index, curr_min_error in enumerate(min_errors_list):
        barplot_sequences.text(index, curr_min_error,
                               round(curr_min_error, 2), ha='center')

    plt.show()


def sequences_num_of_epochs_plot():
    datasets_list = ['factorial', 'fibonacci', 'period', 'exponent']

    min_errors_list = list()

    for curr_dataset in tqdm(datasets_list, postfix=f'Training networks'):
        Config.data_type = curr_dataset

        total_error_list = perform_pipeline_for_plots()

        min_errors_list.append(round(min(total_error_list), 2))

    print(min_errors_list)

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.set_title(f'Sequence vs min error achieved for {Config.num_epochs} epochs')

    ax.set_xlabel('Sequence')
    ax.set_ylabel('Min error')

    barplot_sequences = sns.barplot(x=datasets_list, y=min_errors_list, ax=ax)

    for index, curr_min_error in enumerate(min_errors_list):
        barplot_sequences.text(index, curr_min_error,
                               round(curr_min_error, 2), ha='center')

    plt.show()


if __name__ == '__main__':
    learning_rate_epochs_plot()
    # errors_epochs_plot()
    # sequences_num_of_epochs_plot()
    # number_of_input_values_epochs_plot()

