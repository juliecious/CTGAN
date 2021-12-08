from collections import OrderedDict
import torch
import flwr as fl
from utils import convert_adult_ds, eval_dataset
from ctgan.synthesizers.ctgan import CTGANSynthesizer
from sklearn.model_selection import train_test_split
from ctgan.demo import load_demo


def load_data():
    data = load_demo()
    disc = [
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
    target = 'income'
    _data = convert_adult_ds(data)
    X = _data.drop([target], axis=1)
    y = _data[target]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return data, X_test, y_test, disc, target


def main():
    """ load data, start CTGANClient """
    model = CTGANSynthesizer(epochs=1, cuda=torch.cuda.is_available(), verbose=True)
    data, X_test, y_test, disc, target = load_data()
    model.fit(data, disc)

    class FlwrClient(fl.client.NumPyClient):
        def get_parameters(self):
            return [val.cpu().numpy() for _, val in model._generator.state_dict().items()]

        def set_parameters(self, parameters):
            params_dict = zip(model._generator.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            model._generator.load_state_dict(state_dict, strict=False)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            model.fit(model.sample(len(data)), disc)
            return self.get_parameters(), len(data), {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            test_data = model.sample(len(data))
            _samples = convert_adult_ds(test_data)
            X_syn = _samples.drop([target], axis=1)
            y_syn = _samples[target]
            avg_acc = eval_dataset(X_syn, y_syn, X_test, y_test)

            return float(avg_acc), len(test_data), {}

    fl.client.start_numpy_client("[::]:8080", client=FlwrClient())


if __name__ == "__main__":
    main()
