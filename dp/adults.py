import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action='ignore', category=ConvergenceWarning)

from ctgan import load_demo, CTGANSynthesizer, DPCTGANSynthesizer, ADPCTGANSynthesizer, \
    TVAESynthesizer
from sklearn.model_selection import train_test_split
from utils import convert_adult_ds, eval_dataset, plot_scores


if __name__ == '__main__':

    data = load_demo()
    target = 'income'

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

    ctgan = CTGANSynthesizer(cuda=False)
    ctgan.fit(data, discrete_columns)

    dpctgan = DPCTGANSynthesizer(verbose=True, sigma=8, target_epsilon=2, cuda=False)
    print(dpctgan.get_config())
    dpctgan.fit(data, discrete_columns)
    dpctgan.plot_losses()


    # # evaluate performance using real data
    _data = convert_adult_ds(data)
    X = _data.drop([target], axis=1)
    y = _data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print('\nTrain on real, test on real')
    real, trtr = eval_dataset(X_train, y_train, X_test, y_test)

    # evaluate performance using fake data
    # # TVAE
    # samples = tvae.sample(len(data)) # Synthetic copy
    # _samples = convert_adult_ds(samples)
    # X_syn = _samples.drop([target], axis=1)
    # y_syn = _samples[target]
    # print('\nTVAE: Train on fake, test on real')
    # fake_tvae, tstr_tvae = eval_dataset(X_syn, y_syn, X_test, y_test)

    # # CTGAN
    samples = ctgan.sample(len(data)) # Synthetic copy
    _samples = convert_adult_ds(samples)
    X_syn = _samples.drop([target], axis=1)
    y_syn = _samples[target]
    print('\nCTGAN: Train on fake, test on real')
    fake_ctgan, tstr_ctgan = eval_dataset(X_syn, y_syn, X_test, y_test)

    # DPCTGAN
    samples = dpctgan.sample(len(data))  # Synthetic copy
    _samples = convert_adult_ds(samples)
    X_syn = _samples.drop([target], axis=1)
    y_syn = _samples[target]
    print('\nDPCTGAN: Train on fake, test on real')
    fake_dpctgan, tstr_dpctgan = eval_dataset(X_syn, y_syn, X_test, y_test)

    # # ADPCTGAN
    # # samples = adpctgan.sample(len(data))  # Synthetic copy
    # # _samples = convert_adult_ds(samples)
    # # X_syn = _samples.drop([target], axis=1)
    # # y_syn = _samples[target]
    # # print('\nADPCTGAN: Train on fake, test on real')
    # # fake_dpctgan, tstr_adpctgan = eval_dataset(X_syn, y_syn, X_test, y_test)
    #
    # # plot_scores(trtr, tstr)
    # import matplotlib.pyplot as plt
    # import numpy as np
    #
    # metrics = ['acc', 'f1 score', 'auroc', 'auprc']
    # plt.figure(figsize=(10, 5))
    # X = np.arange(4)
    # plt.title("Adult Dataset")
    # plt.bar(X + 0.00, trtr, width=0.25, color='#8FB9AA')
    # plt.bar(X + 0.25, tstr_ctgan, width=0.25, color='#F2D096')
    # plt.bar(X + 0.50, tstr_dpctgan, width=0.25, color='#ED8975')
    # plt.xticks(X + 0.25, metrics)
    # plt.legend(['Real', 'CTGAN', 'DP-CTGAN'], bbox_to_anchor=(1.05, 1), loc='upper left')
    #
    # plt.show()
