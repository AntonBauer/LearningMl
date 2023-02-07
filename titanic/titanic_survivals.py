import os
from typing import Sequence

import matplotlib.pyplot as plt
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


def define_model(input_shape: Sequence[int]) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(4, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['binary_accuracy'])

    return model


def plot_learning_curves(history: tf.keras.callbacks.History):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()


titanic_data = download_dataset(dataset_url, dataset_path)
print_dataset_info(titanic_data, 'Titanic data')
titanic_train, titanic_test = split_dataset(titanic_data, 0.8)

label = 'Survived'
features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
X_train, y_train = split_to_features_and_labels(titanic_data, features, label)

print_dataset_info(X_train, 'Train features')
print_dataset_info(y_train, 'Train labels')

titanic_survivals_model = define_model(X_train.shape[1:])
early_stopping = tf.keras.callbacks.EarlyStopping(
    patience=20, restore_best_weights=True)
history = titanic_survivals_model.fit(
    X_train, y_train,
    epochs=1000,
    validation_split=0.2,
    batch_size=64,
    verbose=0,
    callbacks=[early_stopping])

plot_learning_curves(history)