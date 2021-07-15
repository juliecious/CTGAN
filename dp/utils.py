from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
import numpy as np


def eval_dataset(X, y, X_test, y_test):
    learners = [(AdaBoostClassifier(n_estimators=50)),
                (DecisionTreeClassifier(max_depth=20)),
                (LogisticRegression(max_iter=1000)),
                (MLPClassifier(hidden_layer_sizes=(50,)))]

    history = dict()
    avg_acc, avg_f1, avg_auroc, avg_auprc = 0, 0, 0, 0

    for i in range(len(learners)):
        model = learners[i]
        model.fit(X, y)
        y_score = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_score)
        auprc = average_precision_score(y_test, y_score)

        model_name = str(type(learners[i]).__name__)
        history[model_name] = {'acc': acc, 'f1': f1, 'auroc': auc_score, 'auprc': auprc}
        avg_acc += acc
        avg_f1 += f1
        avg_auroc += auc_score
        avg_auprc += auprc

        # print('-' * 60)
        # print(f'{model_name:<30} acc {round(acc, 4):>5}\t f1 score {round(f1, 4):>5}\t '
        #       f'auroc {round(auc_score, 4):>5}\t auprc {round(auprc, 4):>5}')

    N = len(learners)
    avg_acc, avg_f1, avg_auroc, avg_auprc = avg_acc/N, avg_f1/N, avg_auroc/N, avg_auprc/N
    print(f'Average: acc {round(avg_acc, 4):>5}\t f1 score {round(avg_f1, 4):>5}\t '
          f'auroc {round(avg_auroc, 4):>5}\t auprc {round(avg_auprc, 4):>5}')
    return history


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
