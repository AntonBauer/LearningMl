import os
from typing import List

import opendatasets as od
import pandas as pd
import tensorflow as tf
from pandas import Series


def print_dataset_info(dataset: pd.DataFrame, description: str = ''):
    print(description)
    print(dataset.describe())
    print(dataset.head())
    print()


def download_dataset(url: str, dataset_path: str, files_path: str, file_name: str, print_dataset=False) -> pd.DataFrame:
    od.download(url, dataset_path)
    dataset = pd.read_csv(os.path.join(dataset_path, files_path, file_name))
    if print_dataset:
        print_dataset_info(dataset, 'Titanic data')

    return dataset


def split_into_features_and_label(dataset: pd.DataFrame, label) -> tuple[pd.DataFrame, Series]:
    features = dataset.copy()
    label = features.pop(label)

    return features, label


def drop_unusable_columns(dataset: pd.DataFrame, to_drop: List[str]) -> pd.DataFrame:
    result = dataset.copy()
    for column in to_drop:
        result.pop(column)

    return result
