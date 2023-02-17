import pandas as pd
import tensorflow as tf
import numpy as np

def _to_input_type(column_type):
    if column_type == object:
        return tf.string
    else:
        return tf.float32


def build_symbolic_inputs(dataset: pd.DataFrame) -> dict[str, tf.keras.Input]:
    inputs = {}
    for name, column in dataset.items():
        input_type = _to_input_type(column.dtype)
        inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=input_type)

    return inputs


def extract_numerical_inputs(inputs: dict[str, tf.keras.Input]) -> dict[str, tf.keras.Input]:
    return {name: input for name, input in inputs.items()
            if input.dtype == tf.float32}

def extract_string_inputs(inputs: dict[str, tf.keras.Input]) -> dict[str, tf.keras.Input]:
    return {name: input for name, input in inputs.items()
            if input.dtype != tf.float32}

def normalize_inputs(inputs: dict[str, tf.keras.Input], dataset):
    denormalized = tf.keras.layers.Concatenate(list(inputs.values()))
    normalization_layer = tf.keras.layers.Normalization()
    normalization_layer.adapt(np.array(dataset[inputs.keys()]))

    return normalization_layer(denormalized)

def create_category_encodings(inputs: dict[str, tf.keras.Input], dataset):
    for name, input in inputs.items():
        lookup = tf.keras.layers.StringLookup(vocabulary=np.unique(dataset[name]))
        category_encoding = tf.keras.layers.CategoryEncoding(num_tokens=lookup.vocabulary_size())

        yield category_encoding(lookup(input))

# def create_preprocessing_inputs(dataset: pd.DataFrame):
