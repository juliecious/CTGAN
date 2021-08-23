from collections import OrderedDict
import torch
from sklearn.model_selection import train_test_split

import ctgan
import flwr as fl

from ctgan import CTGANSynthesizer, load_demo
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, \
    f1_score, accuracy_score, log_loss
import numpy as np


def eval_dataset(X, y, X_test, y_test, multiclass=False):
    learners = [(AdaBoostClassifier(n_estimators=50)),
                (DecisionTreeClassifier(max_depth=20)),
                (LogisticRegression(max_iter=1000)),
                (MLPClassifier(hidden_layer_sizes=(50,))),
                # (RandomForestClassifier(random_state=18)),
                # (KNeighborsClassifier(n_neighbors=15)),
                # (XGBClassifier(random_state=18)),
                # (SVC())
                ]

    history = dict()
    avg_acc, avg_f1, avg_auroc, avg_auprc, avg_ll = 0, 0, 0, 0, 0

    if multiclass:
        learners.append((RandomForestClassifier()))
        # print('Multiclass classification metrics:')
    # else:
    #     print('Binary classification metrics:')

    for i in range(len(learners)):
        model = learners[i]
        model.fit(X, y)
        y_pred = model.predict(X_test)

        model_name = str(type(learners[i]).__name__)

        if multiclass:
            y_score = model.predict_proba(X_test)
            # y_score = y_score.reshape(y_test.shape[0])
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            auc_score = roc_auc_score(y_test, y_score, average="weighted", multi_class="ovr")
            ll = log_loss(y_test, y_score)
            history[model_name] = {'acc': acc, 'f1': f1, 'auroc': auc_score, 'log loss': ll}
            avg_acc += acc
            avg_f1 += f1
            avg_auroc += auc_score
            avg_ll += ll

        else:
            y_score = model.predict_proba(X_test)[:, 1]
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc_score = roc_auc_score(y_test, y_score)
            auprc = average_precision_score(y_test, y_score)

            history[model_name] = {'acc': acc, 'f1': f1, 'auroc': auc_score, 'auprc': auprc}
            avg_acc += acc
            avg_f1 += f1
            avg_auroc += auc_score
            avg_auprc += auprc

    N = len(learners)
    avg_acc, avg_f1, avg_auroc, avg_auprc, avg_ll = avg_acc/N, avg_f1/N, avg_auroc/N, avg_auprc/N, avg_ll/N

    if multiclass:
        print(f'Average: acc {round(avg_acc, 4):>5}\t f1 score {round(avg_f1, 4):>5}\t '
              f'auroc {round(avg_auroc, 4):>5}\t log loss {round(avg_ll, 4):>5}')
        return avg_acc, avg_f1, avg_auroc, avg_ll
    else:
        print(f'Average: acc {round(avg_acc, 4):>5}\t f1 score {round(avg_f1, 4):>5}\t '
              f'auroc {round(avg_auroc, 4):>5}\t auprc {round(avg_auprc, 4):>5}')

        return avg_acc, avg_f1, avg_auroc, avg_auprc

def convert_adult_ds(dataset):
    df = dataset.copy()
    salary_map = {' <=50K': 1, ' >50K': 0}
    df['income'] = df['income'].map(salary_map).astype(int)
    df['sex'] = df['sex'].map({' Male': 1, ' Female': 0}).astype(int)
    df['native-country'] = df['native-country'].replace(' ?', np.nan)
    df['workclass'] = df['workclass'].replace(' ?', np.nan)
    df['occupation'] = df['occupation'].replace(' ?', np.nan)
    df.dropna(how='any', inplace=True)

    df.loc[df['native-country'] != ' United-States', 'native-country'] = 'Non-US'
    df.loc[df['native-country'] == ' United-States', 'native-country'] = 'US'
    df['native-country'] = df['native-country'].map({'US': 1, 'Non-US': 0}).astype(int)

    df['marital-status'] = df['marital-status'].replace(
        [' Divorced', ' Married-spouse-absent', ' Never-married', ' Separated',
         ' Widowed'], 'Single')
    df['marital-status'] = df['marital-status'].replace(
        [' Married-AF-spouse', ' Married-civ-spouse'], 'Couple')
    df['marital-status'] = df['marital-status'].map({'Couple': 0, 'Single': 1})
    rel_map = {' Unmarried': 0, ' Wife': 1, ' Husband': 2, ' Not-in-family': 3, ' Own-child': 4,
               ' Other-relative': 5}
    df['relationship'] = df['relationship'].map(rel_map)

    df['race'] = df['race'].map(
        {' White': 0, ' Amer-Indian-Eskimo': 1, ' Asian-Pac-Islander': 2, ' Black': 3,
         ' Other': 4})

    def f(x):
        if x['workclass'] == ' Federal-gov' or x['workclass'] == ' Local-gov' or x[
            'workclass'] == ' State-gov':
            return 'govt'
        elif x['workclass'] == ' Private':
            return 'private'
        elif x['workclass'] == ' Self-emp-inc' or x['workclass'] == ' Self-emp-not-inc':
            return 'self_employed'
        else:
            return 'without_pay'

    df['employment_type'] = df.apply(f, axis=1)
    employment_map = {'govt': 0, 'private': 1, 'self_employed': 2, 'without_pay': 3}
    df['employment_type'] = df['employment_type'].map(employment_map)
    df.drop(labels=['workclass', 'education', 'occupation'], axis=1, inplace=True)

    df.loc[(df['capital-gain'] > 0), 'capital-gain'] = 1
    df.loc[(df['capital-gain'] == 0, 'capital-gain')] = 0
    df.loc[(df['capital-loss'] > 0), 'capital-loss'] = 1
    df.loc[(df['capital-loss'] == 0, 'capital-loss')] = 0

    df.drop(['fnlwgt'], axis=1, inplace=True)
    return df

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

    def get_parameters(self):
        """ Return model parameters as a list of NumPy ndarrays """
        self.model.fit(self.train_data, self.discrete_columns)
        print(self.model._generator is None)
        return [val.cpu().numpy() for name, val in self.model._generator.state_dict().items()
                if 'bn' not in name]

    def set_parameters(self, parameters):
        """ Set generator parameters from a list of NumPy ndarrays """
        self.model.train(self.train_data, self.discrete_columns)
        keys = [k for k in self.model.state_dict().keys() if 'bn' not in k]
        params_dict = zip(keys, parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model._generator.load_state_dict(state_dict, strict=False)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.fit(self.train_data, self.discrete_columns)
        return self.get_parameters(), len(self.train_data), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.test_data = self.model.sample(len(self.train_data))

        _samples = convert_adult_ds(self.test_data)
        X_syn = _samples.drop([self.target], axis=1)
        y_syn = _samples[self.target]

        avg_acc, avg_f1, avg_auroc, avg_auprc = eval_dataset(X_syn, y_syn, self.X_test, self.y_test)
        return float(avg_acc), len(self.test_data), {"accuracy": avg_acc}

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
    target = 'income'
    client = CTGANClient(model, data, discrete_columns, target)
    fl.client.start_numpy_client("0.0.0.0:8080", client)

if __name__ == "__main__":
    main()
