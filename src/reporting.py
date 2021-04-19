
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

import src.plot_utils as pu
from src.classifiers._kmeansbayes import KMeansBayes
from src.utils import get_project_results_dir


def plot_best_k(parameters_dict):
    dataset_name = parameters_dict['dataset_name']
    target_names = parameters_dict['target_names']
    best_k = np.array(parameters_dict['best_k'])

    pu.figure_setup()

    fig_size = pu.get_fig_size(12, 9)
    fig = plt.figure(figsize=(fig_size))
    fig.suptitle(f'Dataset: {dataset_name.upper()}')

    ax = fig.add_subplot()
    ax.set_axisbelow(True)

    for idx in range(len(target_names)):
        x = list(range(1, 11))
        y = best_k[:, idx]
        ax.plot(x, y, label=f'classe \'{target_names[idx]}\'', marker='o')

    ax.set_xlabel('Número da iteração k-Fold')
    ax.set_ylabel('Melhor k (k-Means)')

    plt.legend()
    plt.tight_layout()

    filename = get_project_results_dir().joinpath(
        dataset_name + '_best_k.eps')

    return fig, str(filename)


def plot_acc(parameters_dict):
    measures_lst = parameters_dict['measures_lst']
    dataset_name = parameters_dict['dataset_name']

    pu.figure_setup()

    fig_size = pu.get_fig_size(12, 9)
    fig = plt.figure(figsize=(fig_size))
    fig.suptitle(f'Dataset: {dataset_name.upper()}')

    ax = fig.add_subplot()
    ax.set_axisbelow(True)
    ax.set_xlabel('Classificadores')
    ax.set_ylabel('Acurácia média')
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['k-Means Bayes', '1-NN', 'Bayes'])

    for idx, measure in enumerate(measures_lst):
        acc, acc_std = parameters_dict[measure]
        ax.errorbar(x=idx+1, y=acc, yerr=acc_std,
                    label=f'{measure}', marker='o')

    plt.tight_layout()

    filename = get_project_results_dir().joinpath(
        dataset_name + '_acc.eps')

    return fig, str(filename)


def plot_hq_mtx(parameters_dict):
    measures_lst = parameters_dict['measures_lst']
    dataset_name = parameters_dict['dataset_name']
    target_names = parameters_dict['target_names']
    X = parameters_dict['X']
    y = parameters_dict['y']

    pu.figure_setup()

    fig_size = pu.get_fig_size(15, 4.4)
    fig = plt.figure(figsize=(fig_size))
    fig.suptitle(f'Dataset: {dataset_name.upper()}')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=1, shuffle=True, stratify=y, test_size=0.25)

    for idx, measure in enumerate(measures_lst):
        ax = fig.add_subplot(1, 3, idx+1)
        ax.set_axisbelow(True)
        ax.grid(False)

        if(measure == 'kmb'):
            clf = KMeansBayes(target_names).fit(X_train, y_train)
            title = 'k-Means Bayes'
        elif(measure == '1nn'):
            clf = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)
            title = '1-NN'
        else:
            clf = GaussianNB().fit(X_train, y_train)
            title = 'Bayes'

        cf_mtx = confusion_matrix(y_test, clf.predict(X_test))
        tn, fp, fn, tp = cf_mtx.ravel()

        f_score = tp/(tp + (1/2)*(fn+fp))
        f_score = '{:.3f}'.format(f_score)

        print(
            f'Title:{title} - TN:{tn} FP:{fp} FN:{fn} TP:{tp} F1-measure: {f_score}\n')

        disp = ConfusionMatrixDisplay(confusion_matrix=cf_mtx)

        disp.plot(ax=ax, cmap=plt.cm.Blues)

        ax.set_title(title)

    plt.tight_layout()

    filename = get_project_results_dir().joinpath(dataset_name + '_cf_mtx.eps')

    return fig, str(filename)


def produce_report(parameters_dict):
    fig, filename = plot_best_k(parameters_dict)
    # pu.save_fig(fig, filename)
    fig, filename = plot_acc(parameters_dict)
    # pu.save_fig(fig, filename)
    fig, filename = plot_hq_mtx(parameters_dict)
    # pu.save_fig(fig, filename)
    plt.show()
