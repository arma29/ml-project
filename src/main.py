import src.processing as processing
import src.training as training
import src.reporting as reporting
import time
import src.utils as utils

from src.classifiers._kmeans import KMeansCustom
from sklearn.datasets import make_blobs
import numpy as np



def main():
    start_time = time.time()

    raw_lst = processing.get_data_paths()
    names_lst = processing.get_data_names()

    # for idx in range(len(raw_lst)):
    for idx in range(1):
        data_dict = processing.process_data(raw_lst[idx])
        data_dict['dataset_name'] = names_lst[idx]
        model_dict = training.train_model(data_dict)
        # reporting.produce_report(model_dict)

    utils.print_elapsed_time(time.time() - start_time)


def main2():
    n_samples = 7500
    n_centers = 50
    random_state = 42
    X, y, centers = make_blobs(n_samples=n_samples, n_features=2, centers=n_centers, random_state=random_state, return_centers=True)

    init_lst = ['Rand-P', 'Rand-C', 'Maxmin', 'kmeans++',
                    'Bradley', 'Sorting', 'Projection', 'Luxburg', 'Split']

    magic_number = 1

    global_init = time.time()
    for i in init_lst:
        t_arr = []
        for _ in range(magic_number):
            start_time = time.time()
            # kmn = KMeansCustom(n_clusters=n_centers, init=i,n_init=1, real_centers=centers).fit(X)
            kmn = KMeansCustom(n_clusters=n_centers, init=i,n_init=1, real_centers=centers).initialize_kmeans(X)
            elapsed_time = time.time() - start_time
            t_arr.append(elapsed_time)
        # print(f'CI initial value = {kmn.initial_CI_value} , CI final value = {kmn.final_CI_value}')
        stringed = "{:.3f}".format(np.mean(np.array(t_arr)))
        print(f'Elapsed: {stringed} - Method: {i}')

    stringed = "{:.3f}".format(time.time() - global_init)
    print(f'\nTotal Elapsed: {stringed}')

    # kmn = KMeansCustom(n_clusters=n_centers, init='Luxburg',n_init=1, real_centers=centers).fit(X)
    # print(f'CI initial value = {kmn.initial_CI_value} , CI final value = {kmn.final_CI_value}')


if __name__ == "__main__":
    main2()
