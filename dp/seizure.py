import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action='ignore', category=ConvergenceWarning)

from ctgan import CTGANSynthesizer
from utils import convert_seizure_ds, eval_dataset
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == '__main__':
    url = 'https://raw.githubusercontent.com/juliecious/sml-dataset/master/dataSets/Epileptic_Seizure_Recognition.csv'
    data = pd.read_csv(url)
    data = data.drop('Unnamed', axis=1)
    target = 'y'
    ctgan = CTGANSynthesizer(verbose=True,
                             # epochs=10,
                             private=True,
                             clip_coeff=0.15,
                             # sigma=6,
                             target_epsilon=5,
                             target_delta=1e-5
                             )
    ctgan.fit(data)
    ctgan.plot_losses(save=True)

    # evaluate performance using real data
    data = convert_seizure_ds(data)
    X = data.drop([target], axis=1)
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print('\nTrain on real, test on real')
    real, trtr = eval_dataset(X_train, y_train, X_test, y_test)

    # evaluate performance using fake data
    samples = ctgan.sample(len(data))  # Synthetic copy
    _samples = convert_seizure_ds(samples)
    _samples.dropna(how='any', inplace=True)

    X_syn = _samples.drop([target], axis=1)
    y_syn = _samples[target]
    print('\nTrain on fake, test on real')
    fake, tstr = eval_dataset(X_syn, y_syn, X_test, y_test)

    # plot
    metrics = ['acc', 'f1 score', 'auroc', 'auprc']
    plt.figure(figsize=(5, 5))
    X = np.arange(4)
    plt.title("TRTR v.s. TSTR")
    plt.bar(X + 0.00, trtr, width=0.25)
    plt.bar(X + 0.25, tstr, width=0.25)
    plt.xticks(X + 0.25, metrics)
    plt.legend(['TRTR', 'TSTR'])
    plt.savefig('comparison.png')
    plt.show()

