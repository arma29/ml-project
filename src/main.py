import src.processing as processing
import src.training as training
import src.reporting as reporting
import time
import src.utils as utils

from src.classifiers._kmeans import KMeansCustom
from sklearn.datasets import make_blobs
import numpy as np

from tqdm import tqdm


def main():
    init_methods = [
         'Rand-P',
         'Rand-C',
         'Maxmin',
         'kmeans++',
         #'Bradley',
         'Sorting',
         'Projection',
         'Luxburg',
         'Split',
    ]

    for dataset_name, data, centers in tqdm(list(processing.get_data()), desc="Datasets"):
        for init_method in tqdm(init_methods, desc="Init methods"):
            experiment_data = run_experiment(init_method, data, centers, repetitions=30)
            reporting.produce_report(init_method, dataset_name, experiment_data)


def run_experiment(init_method, data, centers, repetitions=5000):
    experiment_data = []
    n_centers = len(centers)
    for _ in tqdm(range(repetitions), desc="Repetitions"):
        start_time = time.time()
        kmn = KMeansCustom(n_clusters=n_centers, init=init_method, n_init=1, real_centers=centers).fit(data)
        elapsed_time = time.time() - start_time
        experiment_data.append({
            "ci_initial": kmn.initial_CI_value,
            "ci_final": kmn.final_CI_value,
            "elapsed_time": elapsed_time,
            "iterations": kmn.iterations,
        })
    return experiment_data


if __name__ == "__main__":
    main()
