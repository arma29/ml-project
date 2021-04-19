from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import StandardScaler

from src.utils import get_project_data_dir


def get_data_paths():
    raw_path = get_project_data_dir().joinpath('raw')
    files = [join(raw_path, f)
             for f in listdir(raw_path) if isfile(join(raw_path, f))]
    return files


def get_data_names():
    raw_path = get_project_data_dir().joinpath('raw')
    files = [f.split('.')[0]
             for f in listdir(raw_path) if isfile(join(raw_path, f))]
    return files


def process_data(dataset_path):
    data = arff.loadarff(dataset_path)
    df = pd.DataFrame(data[0])

    # Criando o conjunto de treinamento X_ e valores alvo (classes) y_
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.astype('str')

    # Standarization
    standard_scaler = StandardScaler()
    X = standard_scaler.fit_transform(X)

    # Criando conjunto de classes
    if("datatrieve" in dataset_path):
        target_names = np.array(['0', '1'])
    else:  # cm1 is in path
        target_names = np.array(['false', 'true'])

    # Criando dicionário de retorno
    data_dict = {}
    data_dict['X'] = X
    data_dict['y'] = y
    data_dict['target_names'] = target_names

    return data_dict
