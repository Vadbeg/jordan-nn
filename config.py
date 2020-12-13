"""Module with config file"""


class Config:
    """Config class"""

    learning_rate = 0.000003
    momentum = 0.1
    num_epochs = 10_000

    min_error = 0.05
    data_type = 'fibonacci'
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    make_zero_context = False

    num_of_precalculated_values = 12
    num_of_input_elements = 2
    num_of_output_elements = 1

    num_of_hidden_neurons = 7


learning_rate = 0.00003  # коэффициент обучения
num_epochs = 10_000  # кол-во эпох
num_of_precalculated_values = 12  # размер скользящего окна
num_of_input_elements = 1  # кол-во входных значений
num_of_output_elements = 1  # кол-во выходных значений

num_of_hidden_neurons = 7  # кол-во скрытых слоёв

data_type = 'fibonacci'  # последовательность
make_zero_context = False  # режим зануления контекстных нейронов
