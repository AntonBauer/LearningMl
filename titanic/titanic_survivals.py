import helpers
import numpy as np

np.printoptions(precision=3, suppress=True)

dataset_url = 'https://www.kaggle.com/c/titanic/data'
dataset_path = '../data/'
files_path = 'titanic'
train_file_name = 'train.csv'
label_column_name = 'Survived'

titanic_data = helpers.download_dataset(
    dataset_url,
    dataset_path,
    files_path,
    train_file_name,
    print_dataset=True)

titanic_features, titanic_label = helpers.split_into_features_and_label(
    titanic_data,
    label_column_name)

helpers.print_dataset_info(titanic_features, 'Titanic features')
