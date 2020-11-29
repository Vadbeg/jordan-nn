"""Module with training"""

from modules.data.datasets import FactorialDataset
from modules.network.jordan_network import Jordan
from modules.train import train_model
from modules.evaluate import eval_model


if __name__ == '__main__':

    config = {
        'lr': 0.000003,
        'momentum': 0.1,
        'n_epochs': 200_000
    }

    dataset = FactorialDataset(number_of_precalculated_values=5, number_of_input_elements=1)

    in_features = dataset.number_of_input_elements
    out_features = 1

    network = Jordan(lr=config['lr'],
                     momentum=config['momentum'],
                     shape=[in_features, 7, out_features])

    errors_list = train_model(network=network,
                              dataset=dataset,
                              n_epochs=config['n_epochs'])
    accuracy = eval_model(network=network,
                          dataset=dataset)

    print(f'Accuracy: {accuracy}')
