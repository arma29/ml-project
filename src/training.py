import time
from os.path import isfile

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from src.classifiers._kmeansbayes import KMeansBayes
from src.utils import get_project_models_dir


def create_dict(data_dict):
    parameters_dict = {
        'measures_lst': ['kmb', '1nn', 'bayes'],  # cada um terá [acc, acc_std]
        'magic_number': 5,
        'elapsed_time': 0,
        'X': data_dict['X'],
        'y': data_dict['y'],
        'target_names': data_dict['target_names'],
        'dataset_name': data_dict['dataset_name'],
        'best_k': []
    }

    return parameters_dict


def has_saved_model(dataset_name):
    filename = get_project_models_dir().joinpath(dataset_name + '.joblib')

    if(isfile(filename)):
        return True
    else:
        return False


def get_saved_model(dataset_name):
    filename = get_project_models_dir().joinpath(dataset_name + '.joblib')
    return joblib.load(filename=filename)


def save_model(parameters_dict, dataset_name):
    filename = get_project_models_dir().joinpath(dataset_name + '.joblib')
    joblib.dump(value=parameters_dict, filename=filename)


def train_model(data_dict):
    dataset_name = data_dict['dataset_name']
    if(has_saved_model(dataset_name)):
        return get_saved_model(dataset_name)

    parameters_dict = create_dict(data_dict)
    target_names = parameters_dict['target_names']

    exp_time = time.time()

    X = parameters_dict['X']
    y = parameters_dict['y']

    for measure in parameters_dict['measures_lst']:

        skf = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)

        acc_lst = []
        # Para cada rodada, selecionar o melhor k
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            if(measure == 'kmb'):
                clf = KMeansBayes(target_names).fit(X_train, y_train)
                parameters_dict['best_k'].append(clf.best_k)
            elif(measure == '1nn'):
                clf = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)
            else:
                clf = GaussianNB().fit(X_train, y_train)

            acc_lst.append(clf.score(X_test, y_test))

        parameters_dict[measure] = [np.mean(acc_lst), np.std(acc_lst)]

    parameters_dict['elapsed_time'] = time.time() - exp_time

    save_model(parameters_dict, dataset_name)

    return parameters_dict
