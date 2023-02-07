import os
from typing import Sequence

import opendatasets as od
import pandas as pd
import tensorflow as tf

dataset_url = 'https://www.kaggle.com/c/titanic/data'
dataset_path = '../data/'


def download_dataset(url: str, path: str) -> pd.DataFrame:
    od.download(url, path)
    return pd.read_csv(os.path.join(dataset_path, 'titanic', 'train.csv'))


def print_dataset_info(dataset: pd.DataFrame, description: str = ''):
    print(description)
    print(dataset.describe())
    print(dataset.head())
    print()


def split_dataset(dataset: pd.DataFrame, split_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_copy = dataset.copy()
    train_dataset = dataset_copy.sample(frac=split_ratio, random_state=0)
    test_dataset = dataset_copy.drop(train_dataset.index)
    return train_dataset, test_dataset


def split_to_features_and_labels(dataset: pd.DataFrame, features, label) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = dataset[label]
    X = dataset[features]
    return X, y


titanic_data = download_dataset(dataset_url, dataset_path)
print_dataset_info(titanic_data, 'Titanic data')
titanic_train, titanic_test = split_dataset(titanic_data, 0.8)

label = 'Survived'
features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
X_train, y_train = split_to_features_and_labels(titanic_train, features, label)

print_dataset_info(X_train, 'Train features')
print_dataset_info(y_train, 'Train labels')