# Jordan RNN implementation

Implementation of Jordan neural network.

It uses numpy for matrix multiplication. 
Gradient descent algorithm, forward propagation and 
dataset loader are created without usage of
Deep Learning algorithms. Supports only CPU computing.

## Getting Started

To download project:
```
git clone https://github.com/Vadbeg/jordan_nn.git
```


### Installing
To install all libraries you need, print in `autoencoder` directory: 

```
pip install -r requirements.txt
```

It will install all essential libraries


### Config

To use project with `start_training.py` you need to setup config. 
Config is located in `config.py` Config class. Example:

```
class Config:
    learning_rate = 0.0003
    momentum = 0.1
    num_epochs = 200_000

    min_error = 0.05
    dataset = 'fibonacci'

    num_of_precalculated_values = 5
    num_of_input_elements = 1

    num_of_hidden_neurons = 7
``` 


### Usage 

After libraries installation you can ran training and evaluation for different
sequences. All settings for it are settled in `config` dictionaries separately for every sequence. 
Script will produce learning plots for each of the 
sequences:

```
python test.py
```  

Or you can run `start_training.py` script for sequence you want:

```
python start_training.py
```

Sequences are listed below:

* fibonacci
* periodical function (1, 0, -1, 0, 1, 0, -1, ...)
* factorial
* exponential

## Built With

* [numpy](https://flask.palletsprojects.com/en/1.1.x/) - The math framework used.

## Authors

* **Vadim Titko** aka *Vadbeg* - [GitHub](https://github.com/Vadbeg/PythonHomework/commits?author=Vadbeg) 
| [LinkedIn](https://www.linkedin.com/in/vadtitko/)