"""Module with data generators"""

from typing import List, Tuple

from modules.data.base_dataset import BaseDataset


class FibonacciDataset(BaseDataset):
    def __init__(self, max_dataset_length: int = 1000):
        super().__init__()

        self.max_dataset_length = max_dataset_length

        self.data = self.__generate_precalculated_values__()
        self.max_value = max(self.data) + 1
        self.sequence_length = 2

    def __generate_precalculated_values__(self):
        data = list()

        for i in range(self.max_dataset_length + 1):
            if i == 0 or i == 1:
                data.append(1)
            else:
                data.append(data[i - 1] + data[i - 2])

        return data

    def __getitem__(self, item) -> Tuple[List[int], List[int]]:
        input_number_first = self.data[item]
        input_number_second = self.data[item + 1]

        result_number = self.data[item + 2]

        input_mask_first = [0] * self.max_value
        input_mask_first[input_number_first] = 1

        input_mask_second = [0] * self.max_value
        input_mask_second[input_number_second] = 1

        input_mask = input_mask_first + input_mask_second

        output_mask = [0] * self.max_value
        output_mask[result_number] = 1

        return input_mask, output_mask

    def __len__(self):
        # TODO: maybe error is here
        return self.max_dataset_length


class PeriodDataset(BaseDataset):
    def __init__(self, max_dataset_length: int = 1000):
        super().__init__()

        self.max_dataset_length = max_dataset_length

        self.data = self.__generate_precalculated_values__()

        self.ohe_mapped_values = {1: [0, 0, 1],
                                  0: [0, 1, 0],
                                  -1: [1, 0, 0]}
        self.sequence_length = 2

    def __generate_precalculated_values__(self):
        data = list()

        repeated_values = [1, 0, -1, 0]

        for i in range(self.max_dataset_length + 1):
            value_to_insert = repeated_values[i % len(repeated_values)]
            # value_to_insert = 1

            data.append(value_to_insert)

        return data

    def __getitem__(self, item) -> Tuple[List[int], List[int]]:
        input_number_first = self.data[item]
        input_number_second = self.data[item + 1]

        result_number = self.data[item + 2]

        input_mask_first = self.ohe_mapped_values[input_number_first]
        input_mask_second = self.ohe_mapped_values[input_number_second]

        input_mask = input_mask_first + input_mask_second

        output_mask = self.ohe_mapped_values[result_number]

        return input_mask, output_mask

    def __len__(self):
        # TODO: maybe error is here
        return self.max_dataset_length


class FactorialDataset(BaseDataset):
    def __init__(self, max_dataset_length: int = 1000):
        super().__init__()

        self.max_dataset_length = max_dataset_length

        self.data = self.__generate_precalculated_values__()
        self.max_value = max(self.data) + 1

    def __generate_precalculated_values__(self):
        data = list()

        for i in range(self.max_dataset_length + 1):
            if i == 0 or i == 1:
                data.append(1)
            else:
                data.append(i * data[i - 1])

        return data

    def __getitem__(self, item) -> Tuple[List[int], List[int]]:
        input_number = self.data[item]
        result_number = self.data[item + 1]

        input_mask = [0] * self.max_value
        input_mask[input_number] = 1

        output_mask = [0] * self.max_value
        output_mask[result_number] = 1

        return input_mask, output_mask

    def __len__(self):
        # TODO: maybe error is here
        return self.max_dataset_length


class ExponentialDataset(BaseDataset):
    def __init__(self, max_dataset_length: int = 1000):
        super().__init__()

        self.max_dataset_length = max_dataset_length

        self.data = self.__generate_precalculated_values__()
        self.max_value = max(self.data) + 1

    def __generate_precalculated_values__(self):
        data = list()

        for i in range(self.max_dataset_length + 1):
            value_to_insert = i ** 2 if i % 2 == 0 else i
            data.append(value_to_insert)

        return data

    def __getitem__(self, item) -> Tuple[List[int], List[int]]:
        input_number = self.data[item]
        result_number = self.data[item + 1]

        input_mask = [0] * self.max_value
        input_mask[input_number] = 1

        output_mask = [0] * self.max_value
        output_mask[result_number] = 1

        return input_mask, output_mask

    def __len__(self):
        # TODO: maybe error is here
        return self.max_dataset_length


if __name__ == '__main__':
    fibonacci_dataset = FibonacciDataset(max_dataset_length=20)

    for idx, element in enumerate(fibonacci_dataset):
        number_first = element[0][:fibonacci_dataset.max_value].index(1)
        number_second = element[0][fibonacci_dataset.max_value:].index(1)
        second = element[1].index(1)

        print(f'First: {number_first, number_second}. Second: {second}')
