import time

import numpy as np
from sklearn.datasets import make_blobs
from tqdm import tqdm

import src.processing as processing
import src.reporting as reporting
from src.classifiers._kmeans import KMeansCustom
from src.reporting import show_plots


def produce_reports():
    init_methods = [
        'Rand-P',
        'Rand-C',
        'Maxmin',
        'kmeans++',
        'Bradley',
        'Sorting',
        'Projection',
        'Luxburg',
        'Split',
    ]
    main_datasets = [
        'a1',
        'a2',
        'a3',
        'b1',
        'b2',
        'dim32',
        's1',
        's2',
        's3',
        's4',
        'unb',
    ]
    for dataset_name, data, centers in tqdm(list(processing.get_data()), desc="Datasets"):
        if dataset_name not in main_datasets:
            continue
        for init_method in tqdm(init_methods, desc="Init methods"):
            experiment_data = run_experiment(
                init_method, data, centers, repetitions=5000)
            reporting.produce_report(
                init_method, dataset_name, experiment_data)


def run_experiment(init_method, data, centers, repetitions=5000):
    experiment_data = []
    n_centers = len(centers)
    for _ in tqdm(range(repetitions), desc="Repetitions"):
        start_time = time.time()
        kmn = KMeansCustom(n_clusters=n_centers, init=init_method,
                           n_init=1, real_centers=centers).fit(data)
        elapsed_time = time.time() - start_time
        experiment_data.append({
            "ci_initial": kmn.initial_CI_value,
            "ci_final": kmn.final_CI_value,
            "elapsed_time": elapsed_time,
            "iterations": kmn.n_iter_,
        })
    return experiment_data


def run_simple_experiment():
    n_samples = 7500
    n_centers = 50
    random_state = 170
    X, y, centers = make_blobs(n_samples=n_samples, n_features=2,
                               centers=n_centers, random_state=random_state, return_centers=True)

    init_lst = ['Rand-P', 'Rand-C', 'Maxmin', 'kmeans++',
                'Bradley', 'Sorting', 'Projection', 'Luxburg', 'Split']

    magic_number = 5
    for i in init_lst:
        time_arr = []
        for _ in range(magic_number):
            start_time = time.time()
            kmn = KMeansCustom(n_clusters=n_centers, init=i,
                               n_init=1, real_centers=centers).fit(X)
            time_arr.append(time.time() - start_time)
        stringed = '{:.3f}'.format(np.mean(np.array(time_arr)))
        print(f'Elapsed time: {stringed} - Method: {i}')
        print(f'N_iter = {kmn.n_iter_}, Inertia = {kmn.inertia_}, CI initial value = {kmn.initial_CI_value} , CI final value = {kmn.final_CI_value}\n')


def main():
    # produce_reports()
    run_simple_experiment()
    show_plots()


if __name__ == "__main__":
    main()
