import numpy as np
import pandas as pd
import tensorflow as tf


def _to_input_type(column_type):
    if column_type == object:
        return tf.string
    else:
        return tf.float32


def _build_symbolic_inputs(dataset: pd.DataFrame) -> dict[str, tf.keras.Input]:
    inputs = {}
    for name, column in dataset.items():
        input_type = _to_input_type(column.dtype)
        inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=input_type)

    return inputs


def _extract_numerical_inputs(inputs: dict[str, tf.keras.Input]) -> dict[str, tf.keras.Input]:
    return {name: input for name, input in inputs.items()
            if input.dtype == tf.float32}


def _extract_string_inputs(inputs: dict[str, tf.keras.Input]) -> dict[str, tf.keras.Input]:
    return {name: input for name, input in inputs.items()
            if input.dtype != tf.float32}


def _normalize_inputs(inputs: dict[str, tf.keras.Input], dataset):
    denormalized = tf.keras.layers.Concatenate(list(inputs.values()))
    normalization_layer = tf.keras.layers.Normalization()
    normalization_layer.adapt(np.array(dataset[inputs.keys()]))

    return normalization_layer(denormalized)


def _create_category_encodings(inputs: dict[str, tf.keras.Input], dataset):
    for name, input in inputs.items():
        lookup = tf.keras.layers.StringLookup(
            vocabulary=np.unique(dataset[name]))
        category_encoding = tf.keras.layers.CategoryEncoding(
            num_tokens=lookup.vocabulary_size())

        yield category_encoding(lookup(input))


def create_preprocessing_head(dataset: pd.DataFrame) -> tf.keras.Model:
    symbolic_inputs = _build_symbolic_inputs(dataset)
    numerical_inputs = _extract_numerical_inputs(symbolic_inputs)
    string_inputs = _extract_string_inputs(symbolic_inputs)

    normalized_layer = _normalize_inputs(numerical_inputs, dataset)
    category_encodings_layer = list(
        _create_category_encodings(string_inputs, dataset))

    preprocessing_inputs = tf.keras.layers.Concatenate()(normalized_layer + category_encodings_layer)
    return tf.keras.Model(symbolic_inputs, preprocessing_inputs)
