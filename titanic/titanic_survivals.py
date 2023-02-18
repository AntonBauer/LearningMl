import numpy as np

from dataset_helpers import *
from input_helpers import *

np.printoptions(precision=3, suppress=True)

dataset_url = 'https://www.kaggle.com/c/titanic/data'
dataset_path = '../data/'
files_path = 'titanic'
train_file_name = 'train.csv'
label_column_name = 'Survived'

titanic_data = download_dataset(
    dataset_url,
    dataset_path,
    files_path,
    train_file_name,
    print_dataset=True)

titanic_cleared = drop_unusable_columns(
    titanic_data,
    ['PassengerId', 'Name', 'Embarked', 'Ticket'])

titanic_features, titanic_label = split_into_features_and_label(
    titanic_cleared,
    label_column_name)

print_dataset_info(titanic_features, 'Titanic features')

preprocessing_head = create_preprocessing_head(titanic_features)
tf.keras.utils.plot_model(model=preprocessing_head, rankdir='LR', dpi=72, show_shapes=True)
