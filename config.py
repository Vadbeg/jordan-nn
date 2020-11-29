"""Module with config file"""


class Config:
    """Config class"""

    learning_rate = 0.0003
    momentum = 0.1
    num_epochs = 200_000

    min_error = 0.05
    dataset = 'fibonacci'

    num_of_precalculated_values = 5
    num_of_input_elements = 1

    num_of_hidden_neurons = 7
