import os

import opendatasets as od
import pandas as pd
from pandas import Series


def print_dataset_info(dataset: pd.DataFrame, description: str = ''):
    print(description)
    print(dataset.describe())
    print(dataset.head())
    print()


def download_dataset(url: str, dataset_path: str, print_dataset=False) -> pd.DataFrame:
    od.download(url, dataset_path)
    dataset = pd.read_csv(os.path.join(dataset_path, 'titanic', 'train.csv'))
    if print_dataset:
        print_dataset_info(dataset, 'Titanic data')

    return dataset


def split_into_features_and_label(dataset: pd.DataFrame, label) -> tuple[pd.DataFrame, Series]:
    features = dataset.copy()
    label = features.pop(label)

    return features, label
