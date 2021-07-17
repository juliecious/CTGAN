import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action='ignore', category=ConvergenceWarning)

from ctgan import CTGANSynthesizer, load_demo
from sklearn.model_selection import train_test_split
from utils import convert_adult_ds, eval_dataset


if __name__ == '__main__':

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

    ctgan = CTGANSynthesizer(epochs=500, verbose=True, private=True)
    ctgan.fit(data, discrete_columns)
    ctgan.plot_losses(save=True)

    # evaluate performance using real data
    _data = convert_adult_ds(data)
    X = _data.drop(['income'], axis=1)
    y = _data['income']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print('\nTrain on real, test on real')
    real = eval_dataset(X_train, y_train, X_test, y_test)

    # evaluate performance using fake data
    samples = ctgan.sample(len(data)) # Synthetic copy
    _samples = convert_adult_ds(samples)
    X_syn = _samples.drop(['income'], axis=1)
    y_syn = _samples['income']
    print('\nTrain on fake, test on real')
    fake = eval_dataset(X_syn, y_syn, X_test, y_test)

