import warnings
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from ctgan.synthesizers.dp_ctgan import DPCTGANSynthesizer
from sklearn.model_selection import train_test_split
from utils import eval_dataset, plot_scores

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=ConvergenceWarning)


# https://www.kaggle.com/dishrah/creditcard-fraud-detection
if __name__ == '__main__':
    data = pd.read_csv('../examples/csv/creditcard.csv')
    target = 'Class'

    ctgan = DPCTGANSynthesizer(verbose=True,
                               private=True,
                               clip_coeff=0.1,
                               sigma=1,
                               target_delta=1e-5,
                               target_epsilon=1
                               )
    ctgan.fit(data)
    ctgan.plot_losses(save=False)

    X = data.drop([target], axis=1)
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print('\nTrain on real, test on real')
    hist_real, trtr = eval_dataset(X_train, y_train, X_test, y_test)

    samples = ctgan.sample(len(data))  # Synthetic copy
    _samples = samples
    X_syn = _samples.drop([target], axis=1)
    y_syn = _samples[target]

    # rf = RandomForestClassifier(random_state=18)
    # rf.fit(X_syn, y_syn)
    # y_pred = rf.predict(X_test)
    # y_score = rf.predict_proba(X_test)
    # y_score = y_score.reshape(y_test.shape[0])
    # print(y_score.shape, y_score)
    print('\nTrain on fake, test on real')

    hist_fake, tstr = eval_dataset(X_syn, y_syn, X_test, y_test, multiclass=True)
    #
    plot_scores(trtr, tstr)


