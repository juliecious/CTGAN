import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action='ignore', category=ConvergenceWarning)

from ctgan import CTGANSynthesizer
from ctgan.synthesizers.dp_ctgan import DPCTGANSynthesizer
from utils import convert_cervical_ds, eval_dataset
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == '__main__':
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00383/risk_factors_cervical_cancer.csv'
    target = 'Biopsy'
    data = pd.read_csv(url)
    num_cols = data._get_numeric_data().columns
    cols = data.columns
    discrete_columns = list(set(cols) - set(num_cols))
    print(discrete_columns)
    # ctgan = CTGANSynthesizer(epochs=10)
    # ctgan.fit(data, discrete_columns)
    #
    # dpctgan = DPCTGANSynthesizer(verbose=True,
    #                          # epochs=10,
    #                          clip_coeff=0.1,
    #                          sigma=8,
    #                          target_epsilon=2,
    #                          target_delta=1e-5
    #                          )
    # dpctgan.fit(data, discrete_columns)
    # dpctgan.plot_losses()
    #
    # # evaluate performance using real data
    # data = convert_cervical_ds(data)
    # X = data.drop([target], axis=1)
    # y = data[target]
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # print('\nTrain on real, test on real')
    # real, trtr = eval_dataset(X_train, y_train, X_test, y_test)
    #
    # # evaluate performance using fake data
    # # CTGAN
    # samples = ctgan.sample(len(data))  # Synthetic copy
    # _samples = convert_cervical_ds(samples)
    #
    # X_syn = _samples.drop([target], axis=1)
    # y_syn = _samples[target]
    # print('\nCTGAN: Train on fake, test on real')
    # fake_ctgan, tstr_ctgan = eval_dataset(X_syn, y_syn, X_test, y_test)
    #
    # # DPCTGAN
    # samples = dpctgan.sample(len(data))  # Synthetic copy
    # _samples = convert_cervical_ds(samples)
    # X_syn = _samples.drop([target], axis=1)
    # y_syn = _samples[target]
    # print('\nDPCTGAN: Train on fake, test on real')
    # fake_dpctgan, tstr_dpctgan = eval_dataset(X_syn, y_syn, X_test, y_test)
    #
    # # plot
    # metrics = ['acc', 'f1 score', 'auroc', 'auprc']
    # plt.figure(figsize=(10, 5))
    # X = np.arange(4)
    # plt.title("Cervical Dataset")
    # plt.bar(X + 0.00, trtr, width=0.25, color='#8FB9AA')
    # plt.bar(X + 0.25, tstr_ctgan, width=0.25, color='#F2D096')
    # plt.bar(X + 0.50, tstr_dpctgan, width=0.25, color='#ED8975')
    # plt.xticks(X + 0.25, metrics)
    # plt.legend(['Real', 'CTGAN', 'DP-CTGAN'])
    #
    # plt.show()
