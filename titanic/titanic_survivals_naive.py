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
    validation_dataset = dataset_copy.drop(train_dataset.index)
    return train_dataset, validation_dataset


def split_to_features_and_labels(dataset: pd.DataFrame, features, label) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = dataset[label]
    X = dataset[features]
    return X, y


def define_model(input_shape: Sequence[int]) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['binary_accuracy'])

    return model


def plot_learning_curves(history: tf.keras.callbacks.History):
    epochs = range(1, len(history.history['loss']) + 1)
    accuracy = history.history['binary_accuracy']
    validation_accuracy = history.history['val_binary_accuracy']

    plt.plot(epochs, accuracy, validation_accuracy)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


titanic_data = download_dataset(dataset_url, dataset_path)
print_dataset_info(titanic_data, 'Titanic data')
titanic_train, titanic_validation = split_dataset(titanic_data, 0.8)

label = 'Survived'
features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
X_train, y_train = split_to_features_and_labels(titanic_data, features, label)
X_val, y_val = split_to_features_and_labels(
    titanic_validation, features, label)

print_dataset_info(X_train, 'Train features')
print_dataset_info(y_train, 'Train labels')

titanic_survivals_model = define_model(X_train.shape[1:])
early_stopping = tf.keras.callbacks.EarlyStopping(
    patience=20, restore_best_weights=True)

history = titanic_survivals_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10)

plot_learning_curves(history)
