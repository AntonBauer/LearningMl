import os

import opendatasets as od
import pandas as pd
import tensorflow as tf

dataset_url = 'https://www.kaggle.com/c/titanic/data'
dataset_path = '../data/'

def download_dataset(url: str, path: str) -> pd.DataFrame:
    od.download(url, path)
    return pd.read_csv(os.path.join(dataset_path, 'titanic', 'train.csv')) 


def print_dataset_info(dataset: pd.DataFrame):
    print(dataset.describe())
    print(dataset.head())

def split_dataset(dataset: pd.DataFrame, split_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_copy = dataset.copy()
    train_dataset = dataset_copy.sample(frac=split_ratio, random_state=0)
    test_dataset = dataset_copy.drop(train_dataset.index)
    return train_dataset, test_dataset

titanic_data = download_dataset(dataset_url, dataset_path)
print_dataset_info(titanic_data)
titanic_train, titanic_test = split_dataset(titanic_data, 0.8)
