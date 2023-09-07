# train_model_package

The repository is a repository for the pip package which can help in training deep learning models for image classification tasks in PyTorch example dogs and cats.

The dataset used in this project is taken from Kaggle, The link to the dataset I used https://www.kaggle.com/datasets/tongpython/cat-and-dog

in this repository the above mentioned data of dog and cat is added in data named folder.

to use this as example i copied some and created a small dataset which is considered as sample data folder named as data_1.

To install this pip package on your system you must have Python 3.8 or more than 3.8. and PyTorch library, Use a virtual environment in order to avoid conflicts with matching dependencies you already have.

to install enter the root folder where setup.py file exists, open terminal and use these steps:
1. use command cd imageclassification pip install . or pip install imageclassification 
2. After Successful installation Check the installation pip show imageclassification
3. then import the package.
4. to check the version with import imageclassification print("version", imageclassification.version)

and to use, see the example file main_classification.py this will give you idea how to use the package.
