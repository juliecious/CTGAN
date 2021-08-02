import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action='ignore', category=ConvergenceWarning)

from ctgan.synthesizers.dp_ctgan import DPCTGANSynthesizer
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from utils import eval_dataset

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


if __name__ == '__main__':

    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    target = 'target'
    data = df

    ctgan = DPCTGANSynthesizer(verbose=True,
                             # epochs=10,
                             private=True,
                             clip_coeff=0.1,
                             sigma=4,
                             target_epsilon=7,
                             target_delta=1e-5
                             )
    ctgan.fit(data)
    ctgan.plot_losses(save=True)

    # evaluate performance using real data
    X = data.drop([target], axis=1)
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print('\nTrain on real, test on real')
    real, trtr = eval_dataset(X_train, y_train, X_test, y_test)

    # evaluate performance using fake data
    samples = ctgan.sample(len(data))  # Synthetic copy
    samples.dropna(how='any', inplace=True)

    X_syn = samples.drop([target], axis=1)
    y_syn = samples[target]
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
