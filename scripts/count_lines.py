# Copyright (c) 2024 Alessio Borgi. All rights reserved.


# Function to build a dictionary based on the second information of each line in val_annotations.txt
def count_lines_per_class(val_annotations_file):
    '''Builds a dictionary based on the second information of each line in val_annotations.txt.
    
    Args:
    val_annotations_file (str): Path to the val_annotations.txt file.
    
    Returns:
    dict: A dictionary where keys are class labels and values are the number of lines associated with each class.
    '''
    class_count = {}
    with open(val_annotations_file, 'r') as annotations_file:
        for line in annotations_file:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                label = parts[1]
                class_count[label] = class_count.get(label, 0) + 1
    return class_count

# Define the path to val_annotations.txt
val_annotations_file = "./nano-imagenet-30/nano-imagenet-30/val/val_annotations.txt"

# Build the dictionary and count the number of lines for each class
class_count = count_lines_per_class(val_annotations_file)

# Print class counts
sum_lines = 0
sum_classes = 0
for label, count in class_count.items():
    print(f"Class {label}: {count} lines")
    sum_classes += 1
    sum_lines += count

print("The sum of lines is: ", sum_lines)
print("The sum of classes is: ", sum_classes)
