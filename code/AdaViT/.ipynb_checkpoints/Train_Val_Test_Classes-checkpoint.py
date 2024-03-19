
# Importing PyTorch-related Libraries.
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToPILImage
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts
from torchmetrics.classification import Accuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall

# Importing PyTorch Lightning-Related Libraries.
import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import TQDMProgressBar, LearningRateMonitor, ModelCheckpoint

# Importing General Libraries.
import os
import csv
import PIL
import time
import math
import glob
import random
import numpy as np
from PIL import Image
import seaborn as sns
from pathlib import Path
from scipy.stats import norm
import matplotlib.pyplot as plt
from collections import OrderedDict



class CustomTrainingTinyImagenet(ImageFolder):

    def __init__(self, root, transform=None):
        """
        Custom dataset class for Tiny ImageNet Training data.

        Args:
        - root (str): Root directory containing the dataset.
        - transform (callable, optional): Optional transform to be applied to the Input Image.
        """
        super(CustomTrainingTinyImagenet, self).__init__(root, transform=transform)

        # Create mappings between class labels and numerical indices
        self.class_to_index = {cls: idx for idx, cls in enumerate(sorted(self.classes))}
        self.index_to_class = {idx: cls for cls, idx in self.class_to_index.items()}

    def __getitem__(self, index):
        """
        Method to retrieve an item from the dataset.

        Args:
        - index (int): Index of the item to retrieve.

        Returns:
        - sample (torch.Tensor): Transformed image sample.
        - target (int): Numerical index corresponding to the class label.
        """
        # Retrieve the item and its label from the Dataset.
        path, target = self.samples[index]

        # Load the image using the default loader.
        sample = self.loader(path)

        # Apply the specified transformations, if any.
        if self.transform is not None:
            sample = self.transform(sample)

        # Adjust the directory depth to get the target label.
        target_str = os.path.basename(os.path.dirname(os.path.dirname(path)))

        # Convert string label to numerical index using the mapping.
        target = self.class_to_index[target_str]

        return sample, target

    def get_class_from_index(self, index):
        """
        Method to retrieve the class label from a numerical index.

        Args:
        - index (int): Numerical index corresponding to the class label.

        Returns:
        - class_label (str): Class label corresponding to the numerical index.
        """

        return self.index_to_class[index]
    
    


class CustomValidationTinyImagenet(pl.LightningDataModule):

    def __init__(self, root, transform=None):
        """
        Custom data module for Tiny ImageNet Validation data.

        Args:
        - root (str): Root directory containing the dataset.
        - transform (callable, optional): Optional transform to be applied to the Input Image.
        """
        self.root = Path(root)
        self.transform = transform

        # Load and preprocess labels
        self.labels = self.load_labels()
        self.label_to_index = {label: idx for idx, label in enumerate(sorted(set(self.labels.values())))}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}

    def load_labels(self):
        """
        Method to load and Pre-Process Labels from the Validation Dataset.

        Returns:
        - labels (dict): Dictionary mapping image names to labels.
        """
        
        #label_path = "./dataset/pico-imagenet-10/val/val_annotations.txt"
        label_path = os.path.dirname(os.path.dirname(self.root))+"/val_annotations.txt"
        labels = {}

        with open(label_path, "r") as f:
            lines = f.readlines()

        for i,line in enumerate(lines):
            if i == 0:
                parts = line.split("\t")
                image_name, label = parts[0], parts[1]
                labels['val_0.JPEG'] = label
            else:
                parts = line.split("\t")
                image_name, label = parts[0], parts[1]
                labels[image_name] = label

        return labels
    def __len__(self):
        """
        Method to get the length of the dataset.

        Returns:
        - length (int): Number of items in the dataset.
        """
        return len(self.labels)

    def __getitem__(self, index):
        """
        Method to retrieve an item from the dataset.

        Args:
        - index (int): Index of the item to retrieve.

        Returns:
        - image (torch.Tensor): Transformed image sample.
        - label (int): Numerical index corresponding to the class label.
        """
        image_name = f"val_{index}.JPEG"
        image_path = self.root / image_name

        # Open the image using PIL and convert to RGB.
        image = Image.open(image_path).convert("RGB")

        # Apply the specified transformations, if any.
        if self.transform:
            image = self.transform(image)

        # Use the get method to handle cases where the key is not present.
        label_str = self.labels.get(image_name, 'Label not found')

        # Convert string label to numerical index using the mapping.
        label = self.label_to_index[label_str]

        return image, label

    def get_label_from_index(self, index):
        """
        Method to retrieve the class label from a numerical index.

        Args:
        - index (int): Numerical index corresponding to the class label.

        Returns:
        - class_label (str): Class label corresponding to the numerical index.
        """
        return self.index_to_label[index]
    
    

class CustomTestTinyImagenet(pl.LightningDataModule):

    def __init__(self, root, transform=None):
        """
        Custom dataset class for Tiny ImageNet Test data.

        Args:
        - root (str): Root directory containing the dataset.
        - transform (callable, optional): Optional transform to be applied to the Input Image.
        """
        self.root = root
        self.transform = transform
        self.image_paths = self._get_image_paths()

    def __len__(self):
        """
        Method to get the total number of items in the dataset.

        Returns:
        - int: Total number of items in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, index):
        """
        Method to retrieve an item from the dataset.

        Args:
        - index (int): Index of the item to retrieve.

        Returns:
        - sample (torch.Tensor): Transformed image sample.
        - filename (str): Filename of the image.
        """
        # Get the image path based on the index.
        image_path = self.image_paths[index]

        # Load the image using the default loader.
        sample = Image.open(image_path)

        # Apply the specified transformations, if any.
        if self.transform is not None:
            sample = self.transform(sample)

        # Extract the filename from the image path.
        filename = os.path.basename(image_path)

        # Return a tuple containing the sample and filename.
        return sample, filename

    def _get_image_paths(self):
        """
        Helper method to get the paths of all images in the test dataset.

        Returns:
        - list: List of image paths.
        """
        image_paths = [os.path.join(self.root, filename) for filename in os.listdir(self.root)]
        return image_paths




def show_images_labels(images, labels, title):
    """
    Display Images with corresponding Labels.

    Parameters:
    - images (list of tensors): List of Image tensors.
    - labels (list): List of corresponding Labels.
    - title (str): Title for the entire subplot.

    Returns:
    None
    """
    # Create a Subplot with 1 row and len(images) columns.
    fig, axs = plt.subplots(1, len(images), figsize=(8, 4))

    # Set the title for the entire subplot.
    fig.suptitle(title)

    # Iterate over Images and Labels.
    for i, (img, label) in enumerate(zip(images, labels)):
        # Display each Image in a subplot.
        axs[i].imshow(transforms.ToPILImage()(img))

        # Set the title for each subplot with the corresponding label.
        axs[i].set_title(f"Label: {label}")

        # Turn off axis labels for better Visualization.
        axs[i].axis('off')

    # Show the entire subplot.
    plt.show()



def show_images_test(images, title):
    """
    Show a batch of images for testing.

    Parameters:
    - images (list): List of image tensors.
    - title (str): Title of the plot.

    Returns:
    None
    """
    # Create a subplot for each image.
    fig, axs = plt.subplots(1, len(images), figsize=(8, 5))

    # If there's only one image, axs is not iterable, so convert it to a list.
    axs = [axs] if len(images) == 1 else axs

    # Iterate over images.
    for i, (img, ax) in enumerate(zip(images, axs)):
        if isinstance(img, str):
            # If img is a string, load the image using PIL.
            img = Image.open(img)
        ax.imshow(transforms.ToPILImage()(img))  # Assuming images are tensors
        ax.axis('off')

    # Set the title of the plot.
    plt.suptitle(title)
    plt.show()


'''
if __name__ == "__main__":
    # Define the AViT_DataModule.
    data_module = AViT_DataModule(
        train_data_dir="./dataset/pico-imagenet-10/train/",
        val_data_dir="./dataset/pico-imagenet-10/val/images/",
        test_data_dir="./dataset/pico-imagenet-10/test/images/",
        batch_size=32
    )

    # Setup the Dataloaders.
    data_module.setup()

    # Get a batch from the Training DataLoader.
    train_dataloader = data_module.train_dataloader()
    train_batch = next(iter(train_dataloader))

    # Get a batch from the Validation DataLoader.
    val_dataloader = data_module.val_dataloader()
    val_batch = next(iter(val_dataloader))

    # Get a batch from the Test DataLoader.
    test_dataloader = data_module.test_dataloader()
    test_batch = next(iter(test_dataloader))

    # Show two Images from the Training Batch.
    show_images_labels(train_batch[0][:2], train_batch[1][:2], title='Training Batch')

    # Show two Images from the Validation Batch.
    show_images_labels(val_batch[0][:2], val_batch[1][:2], title='Validation Batch')

    # Show two Images from the Test Batch.
    test_batch = next(iter(test_dataloader))
    show_images_test(test_batch[0][:2], title='Test Batch')

'''