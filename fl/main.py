import os
from collections import OrderedDict

import torch
import numpy as np
from sklearn.model_selection import train_test_split

# from ctgan import CTGANSynthesizer
from ctgan import DPCTGANSynthesizer as CTGANSynthesizer
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
N = 3 # client count
target_epsilon = 1

# central server
print('Init central server\n')
ctgan = CTGANSynthesizer(epochs=3, verbose=True, target_epsilon=target_epsilon)
# train
ctgan.fit(data, discrete_columns)
params = [0 for _ in range(len(ctgan._generator.state_dict().keys()))]


""" TODO:
    - need to change the architecture
    - break down the dataset so each client uses different data
"""
# distribute generator's state_dict
cmp_shape = [val.cpu().numpy().shape[0] \
             for val in ctgan._generator.state_dict().values() if val.cpu().numpy().shape]
print(cmp_shape)
for i in range(N):
    client = CTGANSynthesizer(epochs=3, verbose=True, target_epsilon=target_epsilon)
    client._generator = ctgan._generator
    print('...loading server state_dict(). Start training')
    client.fit(data, discrete_columns)
    client.plot_losses(save=True)
    # client.fit(data, discrete_columns, plot=True, plot_dir=os.getcwd())
    print(f'finish client {i} training')
    SELECTED = 0
    client_shape = [val.cpu().numpy().shape[0] \
                    for val in client._generator.state_dict().values() if val.cpu().numpy().shape]
    # TODO: prevent broadcast different size
    if client_shape == cmp_shape:
        SELECTED += 1
        k = 0
        for key, val in client._generator.state_dict().items():
            p = val.cpu().numpy()
            params[k] += p
            k += 1
        print(f'save client {i} state')
    else:
        print('State_dict shape mismatched. Ditch this client model...')

# averaging
print(f'{SELECTED} clients selected.')
params = [el/SELECTED for el in params]
print(f'\nAveraging params from {SELECTED} clients...')

# send back to server
params_dict = zip(ctgan._generator.state_dict().keys(), params)
state_dict = OrderedDict({k: torch.from_numpy(np.array(v)) for k, v in params_dict})
ctgan._generator.load_state_dict(state_dict, strict=False)
print('Send back to server.')

# Synthetic copy
samples = ctgan.sample(len(data))
print('Sampling from the aggregated model...')

_data = convert_adult_ds(samples)
X = _data.drop(['income'], axis=1)
y = _data['income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print('\nTrain on synthetic, test on real')
eval_dataset(X_train, y_train, X_test, y_test)
