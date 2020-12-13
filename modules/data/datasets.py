"""Module with data generators"""

from typing import List, Tuple

from modules.data.base_dataset import BaseDataset
from modules.data.utils import (generate_exp_values,
                                generate_factorial_values,
                                generate_fibonacci_values,
                                generate_period_values)


class CustomDataset(BaseDataset):
    def __init__(self, data: List[int],
                 number_of_input_elements: int = 2,
                 number_of_output_elements: int = 1):
        super().__init__()

        self.number_of_precalculated_values = len(data)
        self.number_of_input_elements = number_of_input_elements
        self.number_of_output_elements = number_of_output_elements

        self.data = data

    def __getitem__(self, item) -> Tuple[List[int], List[int]]:
        if item + self.number_of_input_elements + self.number_of_output_elements >= len(self.data):
            raise IndexError(f'Index is out of range')

        input_numbers = self.data[item: item + self.number_of_input_elements]
        result_number = self.data[item + self.number_of_input_elements:
                                  item + self.number_of_input_elements + self.number_of_output_elements]

        return input_numbers, result_number

    def __len__(self):
        # TODO: maybe error is here
        return self.number_of_precalculated_values - \
               self.number_of_input_elements - \
               self.number_of_output_elements + 1


if __name__ == '__main__':
    number_of_precalculated_values = 5
    data = generate_period_values(
        number_of_precalculated_values=number_of_precalculated_values
    )

    fibonacci_dataset = CustomDataset(data=data,
                                      number_of_input_elements=4,
                                      number_of_output_elements=1)

    for idx, element in enumerate(fibonacci_dataset):
        input_numbers = element[0]
        result_number = element[1]

        print(f'Input numbers: {input_numbers}. Result: {result_number}')

