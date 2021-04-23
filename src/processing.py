from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import StandardScaler

from src.utils import get_project_data_dir


def get_data():
    raw_path = get_project_data_dir().joinpath('raw')
    for dataset_path in raw_path.glob("*.txt"):
        if "-gt" in str(dataset_path):
            continue

        dataset_name = dataset_path.stem
        data = process_data(dataset_path)

        dataset_centers_path = dataset_path.with_name(f"{dataset_name}-gt.txt")
        centers = process_data(dataset_centers_path)

        yield dataset_name, data, centers


def process_data(dataset_path):
    # TODO: Read and process txt files

    df = pd.read_csv(dataset_path, sep="\s+", header=None)
    return df.values
