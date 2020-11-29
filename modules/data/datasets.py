"""Module with data generators"""

from typing import List, Tuple

from modules.data.base_dataset import BaseDataset


class FibonacciDataset(BaseDataset):
    def __init__(self, number_of_precalculated_values: int = 1000, number_of_input_elements: int = 2):
        super().__init__()

        self.number_of_precalculated_values = number_of_precalculated_values
        self.number_of_input_elements = number_of_input_elements

        self.data = self.__generate_precalculated_values__()
        self.max_value = max(self.data) + 1

    def __generate_precalculated_values__(self):
        if self.number_of_input_elements >= self.number_of_precalculated_values:
            raise ValueError(f'Number of input elements is higher or equal '
                             f'to dataset length ({self.number_of_precalculated_values} <= {self.number_of_input_elements})')

        data = list()

        for i in range(self.number_of_precalculated_values + 1):
            if i == 0 or i == 1:
                data.append(1)
            else:
                data.append(data[i - 1] + data[i - 2])

        return data

    def __getitem__(self, item) -> Tuple[List[int], List[int]]:
        input_numbers = self.data[item: item + self.number_of_input_elements]
        result_number = self.data[item + self.number_of_input_elements]

        return input_numbers, result_number

    def __len__(self):
        # TODO: maybe error is here
        return self.number_of_precalculated_values - self.number_of_input_elements


class PeriodDataset(BaseDataset):
    def __init__(self, number_of_precalculated_values: int = 1000, number_of_input_elements: int = 2):
        super().__init__()

        self.number_of_precalculated_values = number_of_precalculated_values
        self.number_of_input_elements = number_of_input_elements

        self.data = self.__generate_precalculated_values__()
        self.max_value = max(self.data) + 1

    def __generate_precalculated_values__(self):
        data = list()

        repeated_values = [1, 0, -1, 0]

        for i in range(self.number_of_precalculated_values + 1):
            value_to_insert = repeated_values[i % len(repeated_values)]
            # value_to_insert = 1

            data.append(value_to_insert)

        return data

    def __getitem__(self, item) -> Tuple[List[int], List[int]]:
        input_numbers = self.data[item: item + self.number_of_input_elements]
        result_number = self.data[item + self.number_of_input_elements]

        return input_numbers, result_number

    def __len__(self):
        # TODO: maybe error is here
        return self.number_of_precalculated_values - self.number_of_input_elements


class FactorialDataset(BaseDataset):
    def __init__(self, number_of_precalculated_values: int = 1000, number_of_input_elements: int = 2):
        super().__init__()

        self.number_of_precalculated_values = number_of_precalculated_values
        self.number_of_input_elements = number_of_input_elements

        self.data = self.__generate_precalculated_values__()
        self.max_value = max(self.data) + 1

    def __generate_precalculated_values__(self):
        data = list()

        for i in range(self.number_of_precalculated_values + 1):
            if i == 0 or i == 1:
                data.append(1)
            else:
                data.append(i * data[i - 1])

        return data

    def __getitem__(self, item) -> Tuple[List[int], List[int]]:
        input_numbers = self.data[item: item + self.number_of_input_elements]
        result_number = self.data[item + self.number_of_input_elements]

        return input_numbers, result_number

    def __len__(self):
        # TODO: maybe error is here
        return self.number_of_precalculated_values - self.number_of_input_elements


class ExponentialDataset(BaseDataset):
    def __init__(self, number_of_precalculated_values: int = 1000, number_of_input_elements: int = 2):
        super().__init__()

        self.number_of_precalculated_values = number_of_precalculated_values
        self.number_of_input_elements = number_of_input_elements

        self.data = self.__generate_precalculated_values__()
        self.max_value = max(self.data) + 1

    def __generate_precalculated_values__(self):
        data = list()

        for i in range(self.number_of_precalculated_values + 1):
            value_to_insert = i ** 2 if i % 2 == 0 else i
            data.append(value_to_insert)

        return data

    def __getitem__(self, item) -> Tuple[List[int], List[int]]:
        input_numbers = self.data[item: item + self.number_of_input_elements]
        result_number = self.data[item + self.number_of_input_elements]

        return input_numbers, result_number

    def __len__(self):
        # TODO: maybe error is here
        return self.number_of_precalculated_values - self.number_of_input_elements


if __name__ == '__main__':
    fibonacci_dataset = FibonacciDataset(number_of_precalculated_values=20, number_of_input_elements=4)

    for idx, element in enumerate(fibonacci_dataset):
        input_numbers = element[0]
        result_number = element[1]
