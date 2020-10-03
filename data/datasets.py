"""Module with data generators"""

from typing import List, Tuple

from data.base_dataset import BaseDataset


class FibonacciDataset(BaseDataset):
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
                data.append(data[i - 1] + data[i - 2])

        return data

    def __getitem__(self, item) -> Tuple[List[int], int]:
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
        pass

    res = fibonacci_dataset.__getitem__(19)

    print(res)
    print(res[0].index(1))
