"""Module with values calculations"""

from typing import List


def generate_fibonacci_values(number_of_precalculated_values: int) -> List[int]:
    """
    Generates fibonacci values

    :param number_of_precalculated_values: number of values to calculate
    :return: list of calculated values
    """

    data = list()

    for i in range(number_of_precalculated_values + 1):
        if i == 0 or i == 1:
            data.append(1)
        else:
            data.append(data[i - 1] + data[i - 2])

    return data


def generate_period_values(number_of_precalculated_values: int) -> List[int]:
    """
    Generates period values

    :param number_of_precalculated_values: number of values to calculate
    :return: list of calculated values
    """

    data = list()

    repeated_values = [1, 0, -1, 0]

    for i in range(number_of_precalculated_values + 1):
        value_to_insert = repeated_values[i % len(repeated_values)]

        data.append(value_to_insert)

    return data


def generate_factorial_values(number_of_precalculated_values: int) -> List[int]:
    """
    Generates factorial values

    :param number_of_precalculated_values: number of values to calculate
    :return: list of calculated values
    """

    data = list()

    for i in range(number_of_precalculated_values + 1):
        if i == 0 or i == 1:
            data.append(1)
        else:
            data.append(i * data[i - 1])

    return data


def generate_exp_values(number_of_precalculated_values: int) -> List[int]:
    """
    Generates exponential values

    :param number_of_precalculated_values: number of values to calculate
    :return: list of calculated values
    """

    data = list()

    for i in range(number_of_precalculated_values + 1):
        value_to_insert = i ** 2 if i % 2 == 0 else i
        data.append(value_to_insert)

    return data
