# Copyright (c) 2024 Alessio Borgi. All rights reserved.

import os
import shutil

# Function to read labels from train folder
def read_train_labels(train_folder):
    '''Reads labels from the train folder.
    
    Args:
    train_folder (str): Path to the train folder.
    
    Returns:
    set: A set containing unique labels.
    '''
    labels = set()
    train_subfolders = os.listdir(train_folder)
    for folder in train_subfolders:
        labels.add(folder)
    return labels

# Function to remove validation images and corresponding annotations
def filter_validation_data(val_folder, train_labels):
    '''Removes validation images and corresponding annotations not present in the train labels.
    
    Args:
    val_folder (str): Path to the validation folder.
    train_labels (set): Set of labels obtained from the train folder.
    
    Returns:
    dict: A dictionary mapping original image names to new image names.
    '''
    val_images_folder = os.path.join(val_folder, "images")
    val_annotations_file = os.path.join(val_folder, "val_annotations.txt")
    new_images_folder = os.path.join(val_folder, "new_images")
    os.makedirs(new_images_folder, exist_ok=True)

    filtered_annotations = []

    new_image_count = 0
    image_name_mapping = {}
    
    with open(val_annotations_file, 'r') as annotations_file:
        lines_seen = set()  # To keep track of unique lines
        for line in annotations_file:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                image_name, label = parts[0], parts[1]
                # Check if the label is present in the train labels
                if label in train_labels:
                    # If the image is in the validation folder but not in the train labels, copy and rename it
                    if image_name in os.listdir(val_images_folder):
                        new_image_name = f"val_{new_image_count}.JPEG"
                        # Copy the image file to the new folder
                        image_path = os.path.join(val_images_folder, image_name)
                        new_image_path = os.path.join(new_images_folder, new_image_name)
                        shutil.copy(image_path, new_image_path)
                        # Update the corresponding line in annotations
                        new_line = line.replace(image_name, new_image_name)
                        filtered_annotations.append(new_line)
                        image_name_mapping[image_name] = new_image_name
                        new_image_count += 1

        # Add unique lines to filtered_annotations
        filtered_annotations = list(set(filtered_annotations))
        
    # Rewrite the filtered annotations without empty lines and update the annotations file
    with open(val_annotations_file, 'w') as annotations_file:
        annotations_file.write('\n'.join(filtered_annotations))

    return image_name_mapping

# Function to count images in a folder
def count_images(folder):
    '''Counts the number of images in a folder.
    
    Args:
    folder (str): Path to the folder.
    
    Returns:
    int: Number of images.
    '''
    return sum([len(files) for _, _, files in os.walk(folder)])

# Define paths
train_folder = "./nano-imagenet-30/nano-imagenet-30/train"
val_folder = "./nano-imagenet-30/nano-imagenet-30/val"

# Read labels from train folder
train_labels = read_train_labels(train_folder)

# Read original number of validation images and annotations
original_val_image_count = count_images(os.path.join(val_folder, "images"))
original_val_annotations_count = len(open(os.path.join(val_folder, "val_annotations.txt")).readlines())

# Filter validation data and get the image name mapping
image_name_mapping = filter_validation_data(val_folder, train_labels)

# Update the lines in val_annotations.txt with the new image names
updated_image_names = set(image_name_mapping.values())
with open(os.path.join(val_folder, "val_annotations.txt"), 'r+') as annotations_file:
    lines = annotations_file.readlines()
    annotations_file.seek(0)
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            image_name, _ = parts[0], parts[1]
            if image_name in updated_image_names:
                annotations_file.write(line)
    annotations_file.truncate()

    # Sort the lines based on the numerical value of X
    annotations_file.seek(0)
    sorted_lines = sorted(annotations_file.readlines(), key=lambda x: int(x.split('.')[0].split('_')[1]))
    annotations_file.truncate(0)
    annotations_file.write(''.join(sorted_lines))

# Read new number of validation images and annotations
new_val_image_count = count_images(os.path.join(val_folder, "new_images"))
new_val_annotations_count = len(open(os.path.join(val_folder, "val_annotations.txt")).readlines())

# Verify the result
print("Original number of validation images:", original_val_image_count)
print("Original number of validation annotations:", original_val_annotations_count)
print("New number of validation images:", new_val_image_count)
print("New number of validation annotations:", new_val_annotations_count)

if new_val_image_count == new_val_annotations_count:
    print("Filtering process successful: Number of images and annotations match.")
else:
    print("Filtering process failed: Number of images and annotations do not match.")
