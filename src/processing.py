
import pandas as pd

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
    df = pd.read_csv(dataset_path, sep="\s+", header=None)
    return df.values
