import helpers

dataset_url = 'https://www.kaggle.com/c/titanic/data'
dataset_path = '../data/'

titanic_data = helpers.download_dataset(
    dataset_url, dataset_path, print_dataset=True)
titanic_features, titanic_label = helpers.split_into_features_and_label(
    titanic_data, 'Survived')

helpers.print_dataset_info(titanic_features, 'Titanic features')
