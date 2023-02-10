import helpers

dataset_url = 'https://www.kaggle.com/c/titanic/data'
dataset_path = '../data/'

titanic_data = helpers.download_dataset(dataset_url, dataset_path, print_dataset=True)