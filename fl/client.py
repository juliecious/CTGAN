from collections import OrderedDict
import torch
from sklearn.model_selection import train_test_split

import ctgan
import flwr as fl

from ctgan import CTGANSynthesizer, load_demo
from utils import convert_adult_ds, eval_dataset


class CTGANClient(fl.client.NumPyClient):
    """ Flower client implementing CTGAN data generation using PyTorch """

    def __init__(self, model, data, discrete_columns, target):
        self.model = model
        self.train_data = data
        self.discrete_columns = discrete_columns
        self.target = target

        _data = convert_adult_ds(data)
        X = _data.drop([target], axis=1)
        y = _data[target]
        _, self.X_test, _, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)


    def get_params(self):
        """ Return model parameters as a list of NumPy ndarrays """
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_params(self, params):
        """ Set model parameters from a list of NumPy ndarrays """
        params_dict = zip(self.model.state_dict().keys, params)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, params, config):
        self.set_params(params)
        ctgan.fit(self.train_data, self.discrete_columns)
        return self.get_params(), len(self.train_data)

    def evaluate(self, params, config):
        self.set_params(params)
        self.test_data = self.model.sample(len(self.train_data))

        _samples = convert_adult_ds(self.test_data)
        X_syn = _samples.drop([self.target], axis=1)
        y_syn = _samples[self.target]

        eval_dataset(X_syn, y_syn, self.X_test, self.y_test)


def main():
    """ load data, start CTGANClient """

    model = CTGANSynthesizer(epochs=10, cuda=torch.cuda.is_available())

    data = load_demo()
    discrete_columns = [
        'workclass',
        'education',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'native-country',
        'income'
    ]
    client = CTGANClient(model, data, discrete_columns)
    fl.client.start_numpy_client("0.0.0.0:8080", client)

if __name__ == "__main__":
    main()
