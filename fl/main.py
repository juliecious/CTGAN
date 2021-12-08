import os
from collections import OrderedDict

import torch
import numpy as np
from sklearn.model_selection import train_test_split

# from ctgan import CTGANSynthesizer
from ctgan import FLDPCTGANSynthesizer as CTGANSynthesizer
from ctgan import load_demo
from fl.utils import convert_adult_ds, eval_dataset

data = load_demo()

# Names of the columns that are discrete
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

# central server
ctgan = CTGANSynthesizer(epochs=2, train_data=data, discrete_columns=discrete_columns)
# train
ctgan.fit(data, discrete_columns)

params = [0 for _ in range(len(ctgan._generator.state_dict().keys()))]


""" TODO:
    - need to change the architecture
    - break down the dataset so each client uses different data
"""
# distribute generator's state_dict
N = 3 # client count
for i in range(N):
    client = CTGANSynthesizer(verbose=True, epochs=3, train_data=data,
                              discrete_columns=discrete_columns)
    client._generator = ctgan._generator
    print('loading server state_dict(). Start training')
    client.fit(data, discrete_columns)
    client.plot_losses(save=True)
    # client.fit(data, discrete_columns, plot=True, plot_dir=os.getcwd())
    print(f'finish client {i} training')
    k = 0
    for _, val in client._generator.state_dict().items():
        params[k] += val.cpu().numpy()
        k += 1
    print(f'save client {i} state')

# averaging
params = [ el/N for el in params]
print('\nAveraging params...')

# send back to server
params_dict = zip(ctgan._generator.state_dict().keys(), params)
state_dict = OrderedDict({k: torch.from_numpy(np.array(v)) for k, v in params_dict})
ctgan._generator.load_state_dict(state_dict, strict=False)
print('Send back to server.')

# Synthetic copy
samples = ctgan.sample(1000)
print('Sampling from the aggregated model...')

_data = convert_adult_ds(samples)
X = _data.drop(['income'], axis=1)
y = _data['income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print('\nTrain on synthetic, test on real')
eval_dataset(X_train, y_train, X_test, y_test)
