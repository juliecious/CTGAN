import warnings
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning

from ctgan import CTGANSynthesizer
from ctgan.synthesizers.dp_ctgan import DPCTGANSynthesizer
from sklearn.model_selection import train_test_split
from utils import eval_dataset, plot_scores

import matplotlib.pyplot as plt
import numpy as np

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=ConvergenceWarning)


# https://www.kaggle.com/dishrah/creditcard-fraud-detection
if __name__ == '__main__':
    data = pd.read_csv('../examples/csv/creditcard.csv')
    target = 'Class'
    data.dropna(how='any', inplace=True)

    ctgan = CTGANSynthesizer(epochs=3, verbose=True)
    ctgan.fit(data)

    dpctgan = DPCTGANSynthesizer(verbose=True,
                                 clip_coeff=0.1,
                                 sigma=1,
                                 target_delta=1e-5,
                                 target_epsilon=2
                                 )
    dpctgan.fit(data)
    dpctgan.plot_losses()

    X = data.drop([target], axis=1)
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print('\nTrain on real, test on real')
    hist_real, trtr = eval_dataset(X_train, y_train, X_test, y_test)

    # # CTGAN
    samples = ctgan.sample(len(data))  # Synthetic copy
    samples.dropna(how='any', inplace=True)
    samples[target] = np.where(samples[target] > 0, 1, 0)
    samples[target] = samples[target].astype(int)

    X_syn = samples.drop([target], axis=1)
    y_syn = samples[target]
    print('\nCTGAN: Train on fake, test on real')
    fake_ctgan, tstr_ctgan = eval_dataset(X_syn, y_syn, X_test, y_test)

    # DPCTGAN
    samples = dpctgan.sample(len(data))  # Synthetic copy
    samples.dropna(how='any', inplace=True)
    samples[target] = np.where(samples[target] > 0, 1, 0)
    samples[target] = samples[target].astype(int)
    X_syn = samples.drop([target], axis=1)
    y_syn = samples[target]
    print('\nDPCTGAN: Train on fake, test on real')
    fake_dpctgan, tstr_dpctgan = eval_dataset(X_syn, y_syn, X_test, y_test)

    metrics = ['acc', 'f1 score', 'auroc', 'auprc']
    plt.figure(figsize=(10, 5))
    X = np.arange(4)
    plt.title("Credit Card Dataset")
    plt.bar(X + 0.00, trtr, width=0.25, color='#8FB9AA')
    plt.bar(X + 0.25, tstr_ctgan, width=0.25, color='#F2D096')
    plt.bar(X + 0.50, tstr_dpctgan, width=0.25, color='#ED8975')
    plt.xticks(X + 0.25, metrics)
    plt.legend(['Real', 'CTGAN', 'DP-CTGAN'], bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.show()
    # # rf = RandomForestClassifier(random_state=18)
    # # rf.fit(X_syn, y_syn)
    # # y_pred = rf.predict(X_test)
    # # y_score = rf.predict_proba(X_test)
    # # y_score = y_score.reshape(y_test.shape[0])
    # # print(y_score.shape, y_score)
    # print('\nTrain on fake, test on real')
    #
    # hist_fake, tstr = eval_dataset(X_syn, y_syn, X_test, y_test, multiclass=True)
    # #
    # plot_scores(trtr, tstr)


