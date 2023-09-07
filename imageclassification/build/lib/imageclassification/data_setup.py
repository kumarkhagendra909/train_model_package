
"""contains funcationality for creating data loaders
 of PyTorch DataLoader's for image classification.
"""

import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

NUM_WORKS = os.cpu_count()

def create_dataloaders(train_dir: str, test_dir: str,
 transform: transforms.Compose,
 batch_size: int,
 num_workers: int = NUM_WORKS):
 """
 Creating training and testing dataloaders.
 Takes in a training directory and testing directory path and turns them into
 PyTorch Datasets and then into PyTorch DataLoaders.

 Args:
 train_dir: Path to training directory.
 test_dir: Path to testing directory.
 transform: torchvision transforms to perform on training and testing data.
 batch_size: Number of samples per batch in each of the DataLoaders.
 num_workers: An integer for number workers per Dataloaders.

 Returns:
 A tuple of (train_dataloader, test_dataloader, class_names).
 Where class names is a list of the target classes.
 Example usage:
 train_dataloader, test_dataloader, class_names = create_dataloaders(train_dir = path/totrain, test_dir = path/totest, transform = some_transform,batch_size = 32, number_workers = 4)
 """

 # funcationality code of this func
 # Use ImageFolder to create dataset (s)
 train_data = datasets.ImageFolder(train_dir, transform = transform)
 test_data = datasets.ImageFolder(test_dir, transform = transform)

 # get class name | example dog, cat, plan, etc.
 class_names = train_data.classes
 print(f"Class names: {class_names}")

 # Turn image into dataloader
 train_dataloader = DataLoader(train_data,
                               batch_size = batch_size,
                               shuffle = True,
                               num_workers= num_workers,
                               pin_memory=True)
 test_dataloader = DataLoader(test_data,
                              batch_size = batch_size,
                              shuffle = False,
                              num_workers=num_workers,
                              pin_memory=True)

 print("creating_data_loader functionality....")
 print(f"Train data loader {train_dataloader} | Test data loader {test_dataloader} | Label names {class_names}")
 return train_dataloader, test_dataloader, class_names

# print(NUM_WORKS)

