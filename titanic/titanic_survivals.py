import dataset_helpers
import input_helpers
import numpy as np

np.printoptions(precision=3, suppress=True)

dataset_url = 'https://www.kaggle.com/c/titanic/data'
dataset_path = '../data/'
files_path = 'titanic'
train_file_name = 'train.csv'
label_column_name = 'Survived'

titanic_data = dataset_helpers.download_dataset(
    dataset_url,
    dataset_path,
    files_path,
    train_file_name,
    print_dataset=True)

titanic_cleared = dataset_helpers.drop_unusable_columns(
    titanic_data,
    ['PassengerId', 'Name', 'Embarked', 'Ticket'])

titanic_features, titanic_label = dataset_helpers.split_into_features_and_label(
    titanic_cleared,
    label_column_name)

dataset_helpers.print_dataset_info(titanic_features, 'Titanic features')

symbolic_inputs = input_helpers.build_symbolic_inputs(titanic_features)
numerical_inputs = input_helpers.extract_numerical_inputs(symbolic_inputs)
string_inputs = input_helpers.extract_string_inputs(symbolic_inputs)

normalized_layer = input_helpers.normalize_inputs(numerical_inputs, titanic_features)
category_encodings_layer =list(input_helpers.create_category_encodings(string_inputs, titanic_features))

preprocessed_inputs = normalized_layer + category_encodings_layer