from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, \
    f1_score, accuracy_score, log_loss
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier


def eval_dataset(X, y, X_test, y_test, multiclass=False):
    learners = [(AdaBoostClassifier(n_estimators=50)),
                (DecisionTreeClassifier(max_depth=20)),
                (LogisticRegression(max_iter=1000)),
                (MLPClassifier(hidden_layer_sizes=(50,))),
                # (RandomForestClassifier(random_state=18)),
                # (KNeighborsClassifier(n_neighbors=15)),
                # (XGBClassifier(random_state=18)),
                # (SVC())
                ]

    history = dict()
    avg_acc, avg_f1, avg_auroc, avg_auprc, avg_ll = 0, 0, 0, 0, 0

    if multiclass:
        learners.append((RandomForestClassifier()))
        # print('Multiclass classification metrics:')
    # else:
    #     print('Binary classification metrics:')

    for i in range(len(learners)):
        model = learners[i]
        model.fit(X, y)
        y_pred = model.predict(X_test)

        model_name = str(type(learners[i]).__name__)

        if multiclass:
            y_score = model.predict_proba(X_test)
            y_score = y_score.reshape(y_test.shape[0])
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            auc_score = roc_auc_score(y_test, y_score, average="weighted", multi_class="ovr")
            ll = log_loss(y_test, y_score)
            history[model_name] = {'acc': acc, 'f1': f1, 'auroc': auc_score, 'log loss': ll}
            avg_acc += acc
            avg_f1 += f1
            avg_auroc += auc_score
            avg_ll += ll

        else:
            y_score = model.predict_proba(X_test)[:, 1]
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc_score = roc_auc_score(y_test, y_score)
            auprc = average_precision_score(y_test, y_score)

            history[model_name] = {'acc': acc, 'f1': f1, 'auroc': auc_score, 'auprc': auprc}
            avg_acc += acc
            avg_f1 += f1
            avg_auroc += auc_score
            avg_auprc += auprc

    N = len(learners)
    avg_acc, avg_f1, avg_auroc, avg_auprc, avg_ll = avg_acc/N, avg_f1/N, avg_auroc/N, avg_auprc/N, avg_ll/N

    if multiclass:
        print(f'Average: acc {round(avg_acc, 4):>5}\t f1 score {round(avg_f1, 4):>5}\t '
              f'auroc {round(avg_auroc, 4):>5}\t log loss {round(avg_ll, 4):>5}')
        return history, (avg_acc, avg_f1, avg_auroc, avg_ll)
    else:
        print(f'Average: acc {round(avg_acc, 4):>5}\t f1 score {round(avg_f1, 4):>5}\t '
              f'auroc {round(avg_auroc, 4):>5}\t auprc {round(avg_auprc, 4):>5}')

        return history, (avg_acc, avg_f1, avg_auroc, avg_auprc)


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


def convert_cervical_ds(df):
    df = df.replace('?', np.nan)
    df = df.apply(pd.to_numeric)

    # Data Imputation
    # for continuous variable
    df['Number of sexual partners'] = df['Number of sexual partners'].fillna(
        df['Number of sexual partners'].median())
    df['First sexual intercourse'] = df['First sexual intercourse'].fillna(
        df['First sexual intercourse'].median())
    df['Num of pregnancies'] = df['Num of pregnancies'].fillna(df['Num of pregnancies'].median())
    df['Smokes'] = df['Smokes'].fillna(1)
    df['Smokes (years)'] = df['Smokes (years)'].fillna(df['Smokes (years)'].median())
    df['Smokes (packs/year)'] = df['Smokes (packs/year)'].fillna(
        df['Smokes (packs/year)'].median())
    df['Hormonal Contraceptives'] = df['Hormonal Contraceptives'].fillna(1)
    df['Hormonal Contraceptives (years)'] = df['Hormonal Contraceptives (years)'].fillna(
        df['Hormonal Contraceptives (years)'].median())
    df['IUD'] = df['IUD'].fillna(0)  # Under suggestion
    df['IUD (years)'] = df['IUD (years)'].fillna(0)  # Under suggestion
    df['STDs'] = df['STDs'].fillna(1)
    df['STDs (number)'] = df['STDs (number)'].fillna(df['STDs (number)'].median())
    df['STDs:condylomatosis'] = df['STDs:condylomatosis'].fillna(
        df['STDs:condylomatosis'].median())
    df['STDs:cervical condylomatosis'] = df['STDs:cervical condylomatosis'].fillna(
        df['STDs:cervical condylomatosis'].median())
    df['STDs:vaginal condylomatosis'] = df['STDs:vaginal condylomatosis'].fillna(
        df['STDs:vaginal condylomatosis'].median())
    df['STDs:vulvo-perineal condylomatosis'] = df['STDs:vulvo-perineal condylomatosis'].fillna(
        df['STDs:vulvo-perineal condylomatosis'].median())
    df['STDs:syphilis'] = df['STDs:syphilis'].fillna(df['STDs:syphilis'].median())
    df['STDs:pelvic inflammatory disease'] = df['STDs:pelvic inflammatory disease'].fillna(
        df['STDs:pelvic inflammatory disease'].median())
    df['STDs:genital herpes'] = df['STDs:genital herpes'].fillna(
        df['STDs:genital herpes'].median())
    df['STDs:molluscum contagiosum'] = df['STDs:molluscum contagiosum'].fillna(
        df['STDs:molluscum contagiosum'].median())
    df['STDs:AIDS'] = df['STDs:AIDS'].fillna(df['STDs:AIDS'].median())
    df['STDs:HIV'] = df['STDs:HIV'].fillna(df['STDs:HIV'].median())
    df['STDs:Hepatitis B'] = df['STDs:Hepatitis B'].fillna(df['STDs:Hepatitis B'].median())
    df['STDs:HPV'] = df['STDs:HPV'].fillna(df['STDs:HPV'].median())
    df['STDs: Time since first diagnosis'] = df['STDs: Time since first diagnosis'].fillna(
        df['STDs: Time since first diagnosis'].median())
    df['STDs: Time since last diagnosis'] = df['STDs: Time since last diagnosis'].fillna(
        df['STDs: Time since last diagnosis'].median())

    # for categorical variable
    df = pd.get_dummies(data=df, columns=['Smokes', 'Hormonal Contraceptives', 'IUD', 'STDs',
                                          'Dx:Cancer', 'Dx:CIN', 'Dx:HPV', 'Dx', 'Hinselmann',
                                          'Citology', 'Schiller'])
    return df


def convert_seizure_ds(df):
    # df = df.drop('Unnamed', axis=1)
    dic = {5: 0, 4: 0, 3: 0, 2: 0, 1: 1}
    df['y'] = df['y'].map(dic)

    return df


def plot_scores(trtr, tstr, save=False):
    metrics = ['acc', 'f1 score', 'auroc', 'auprc']
    plt.figure(figsize=(5, 5))
    X = np.arange(4)
    plt.title("TRTR v.s. TSTR")
    plt.bar(X + 0.00, trtr, width=0.25)
    plt.bar(X + 0.25, tstr, width=0.25)
    plt.xticks(X + 0.25, metrics)
    plt.legend(['TRTR', 'TSTR'])
    if save:
        plt.savefig('comparison.png')
    plt.show()
