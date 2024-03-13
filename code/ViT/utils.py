# Importing PyTorch-related Libraries.
import torch
from torchvision import datasets


# Importing PyTorch Lightning-Related Libraries.
import pytorch_lightning as pl

# Importing General Libraries.
import os
import random
import numpy as np
import matplotlib.pyplot as plt




def set_device():
    """
    Set the device to be used for training and testing.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device



def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results.

    Arguments:
        - seed {int} : Number of the seed.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    pl.seed_everything(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def load_mapping_dict(file_path):
    """
    Load the mapping dictionary from a file.

    Parameters:
    - file_path (str): Path to the file containing mapping data.

    Returns:
    dict: Mapping dictionary with encoded labels as keys and actual labels as values.
    """
    mapping_dict = {}

    with open(file_path, 'r') as file:
        for line in file:
            tokens = line.strip().split('\t')
            if len(tokens) >= 2:
                encoded_label, actual_label = tokens[0], tokens[1]
                mapping_dict[encoded_label] = actual_label

    return mapping_dict



def visualize_dataset(dataset, mapping_dict, num_samples=10):
    """
    Visualize a random sample of images from a dataset along with their labels.

    Parameters:
    - dataset (torchvision.datasets.ImageFolder): Dataset object.
    - mapping_dict (dict): Mapping dictionary with encoded labels as keys and actual labels as values.
    - num_samples (int): Number of samples to visualize.

    Returns:
    None
    """
    class_names = dataset.classes

    np.random.seed(31)
    plt.figure(figsize=(15, 8))

    for i in range(num_samples):
        index = np.random.randint(len(dataset))
        image, encoded_label = dataset[index]
        actual_label = mapping_dict.get(class_names[encoded_label], "Unknown Label")
        actual_label_trimmed = actual_label[:15] + '...' if len(actual_label) > 15 else actual_label

        plt.subplot(2, 5, i+1)
        plt.imshow(np.array(image))
        plt.title(f"Label: {actual_label_trimmed}", wrap=True)
        plt.axis('off')

    plt.tight_layout()
    plt.show()



def visualize_dataset_with_labels(dataset, mapping_dict, num_samples=10):
    """
    Visualize a random sample of images from a dataset along with their labels.

    Parameters:
    - dataset (torchvision.datasets.ImageFolder): Dataset object.
    - mapping_dict (dict): Mapping dictionary with encoded labels as keys and actual labels as values.
    - num_samples (int): Number of samples to visualize.

    Returns:
    None
    """
    class_names = dataset.classes

    np.random.seed(31)
    plt.figure(figsize=(15, 8))

    for i in range(num_samples):
        index = np.random.randint(len(dataset))
        image, encoded_label = dataset[index]
        actual_label = mapping_dict.get(class_names[encoded_label], "Unknown Label")
        actual_label_trimmed = actual_label[:15] + '...' if len(actual_label) > 15 else actual_label

        plt.subplot(2, 5, i+1)
        plt.imshow(np.array(image))
        plt.title(f"Label: {actual_label_trimmed}", wrap=True)
        plt.axis('off')

    plt.tight_layout()
    plt.show()



'''

# Run the code to visualize the Positional Embeddings.
if __name__ == "__main__":
    
    # Set the seed.
    seed_everything(31)

    # Set the device
    device = set_device()

    # Mapping Dict example, and visualize the dataset with original labels.
    mapping_dict = load_mapping_dict('./dataset/words.txt')
    dataset = datasets.ImageFolder(root="./dataset/pico-imagenet-10/train", transform=None)
    visualize_dataset(dataset, mapping_dict, num_samples=10)


    # Mapping dict example, and visualize the dataset with translated labels.
    mapping_dict = load_mapping_dict('./dataset/words.txt')
    dataset = datasets.ImageFolder(root="./dataset/pico-imagenet-10/train/", transform=None)
    visualize_dataset_with_labels(dataset, mapping_dict, num_samples=10)

'''
