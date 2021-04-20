import src.processing as processing
import src.training as training
import src.reporting as reporting
import time
import src.utils as utils

from src.classifiers._kmeans import KMeansCustom
from sklearn.datasets import make_blobs



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
    n_samples = 898
    n_centers = 10
    random_state = 170
    X, y, centers = make_blobs(n_samples=n_samples, n_features=2, centers=n_centers, random_state=random_state, return_centers=True)

    # init_lst = ['Rand-P', 'Rand-C', 'Maxmin', 'kmeans++',
    #                 'Bradley', 'Sorting', 'Projection', 'Luxburg', 'Split']
    
    init_lst = ['Rand-P', 'Rand-C', 'Maxmin', 'kmeans++','Sorting', 'Projection']
    
    # for i in init_lst:
    #     kmn = KMeansCustom(n_clusters=10, init=i,n_init=10, real_centers=centers).fit(X)
    #     print(f'CI initial value = {kmn.initial_CI_value} , CI final value = {kmn.final_CI_value}')

    kmn = KMeansCustom(n_clusters=n_centers, init='Split',n_init=1, real_centers=centers).fit(X)
    print(f'CI initial value = {kmn.initial_CI_value} , CI final value = {kmn.final_CI_value}')
    for i in range(n_centers):
        print(f'Lbl {i} - elements: {len([lbl for lbl in kmn.labels_ if lbl == i])}')


if __name__ == "__main__":
    main2()
