"""Module with base dataset class"""


class BaseDataset:
    def __init__(self):
        pass

    def __generate_precalculated_values__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
