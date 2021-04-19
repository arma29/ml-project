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
    n_samples = 1000
    n_centers = 10
    random_state = 170
    X, y, centers = make_blobs(n_samples=n_samples, n_features=2, centers=n_centers, random_state=random_state, return_centers=True)

    kmn = KMeansCustom(n_clusters=10, init='Rand-C',n_init=1).fit(X)
    print(f'CI value = {kmn.compute_CI_value(centers)}')




if __name__ == "__main__":
    main2()
