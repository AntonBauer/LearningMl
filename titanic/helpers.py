import os

import opendatasets as od
import pandas as pd
import tensorflow as tf


def print_dataset_info(dataset: pd.DataFrame, description: str = ''):
    print(description)
    print(dataset.describe())
    print(dataset.head())
    print()


def download_dataset(url: str, dataset_path: str, print_dataset=False):
    od.download(url, dataset_path)
    dataset = pd.read_csv(os.path.join(dataset_path, 'titanic', 'train.csv'))
    if print_dataset:
        print_dataset_info(dataset, 'Titanic data')

    return tf.data.Dataset.from_tensor_slices(dict(dataset))