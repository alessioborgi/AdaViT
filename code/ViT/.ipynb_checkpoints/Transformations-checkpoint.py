# Importing PyTorch-related Libraries.
from torchvision import transforms

# Importing General Libraries.
import PIL
import numpy as np

# Simple Transformation Class.
class ViT_Transformations:

    def __init__(self):

        self.transform = transforms.Compose([
            # Resize to 224x224
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=(-15, 15)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, sample):
        # Apply AutoAugment first, then the rest of the transforms.
        return self.transform(sample)



# Transformation Class with Custom Cutout Implementation. 
class ViT_Transformations_With_Cutout:

    def __init__(self):

        # Custom cutout implementation with variations
        self.cutout_params = {
            "num_boxes": 5,
            "max_size": int(0.2 * 64),  # 20% of image size (assuming 224x224)
            "shapes": ["rect", "ellipse"],  # Add circular cutouts
            "min_size": int(0.05 * 64),  # Minimum cutout size
        }

        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=(-15, 15)),
            transforms.RandomAffine(degrees=(-15, 15), translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
            transforms.Lambda(self.apply_random_cutout),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def apply_random_cutout(self, img):
        img = np.array(img)
        # Randomly choose number of boxes based on cutout_params
        num_boxes = np.random.randint(1, self.cutout_params["num_boxes"] + 1)
        for _ in range(num_boxes):
            shape = np.random.choice(self.cutout_params["shapes"])
            mask_size = np.random.randint(self.cutout_params["min_size"], self.cutout_params["max_size"] + 1)
            # Adjust mask creation based on shape
            if shape == "rect":
                y, x = np.random.randint(img.shape[0] - mask_size), np.random.randint(img.shape[1] - mask_size)
                y1, y2 = y, y + mask_size
                x1, x2 = x, x + mask_size
            elif shape == "ellipse":
                center_y, center_x = np.random.randint(img.shape[0]), np.random.randint(img.shape[1])
                mask = np.zeros_like(img[:, :, 0])
                radius_y, radius_x = mask_size // 2, mask_size // 2
                y_squared = (center_y - mask.shape[0] // 2) ** 2
                x_squared = (center_x - mask.shape[1] // 2) ** 2
                mask = np.where(y_squared / radius_y**2 + x_squared / radius_x**2 <= 1, 1, 0)
                y1, y2 = max(0, center_y - radius_y), min(img.shape[0], center_y + radius_y + 1)
                x1, x2 = max(0, center_x - radius_x), min(img.shape[1], center_x + radius_x + 1)
            img[y1:y2, x1:x2, :] = 0
        return PIL.Image.fromarray(img)

    def __call__(self, sample):
        # Apply AutoAugment first, then the rest of the transforms
        return self.transform(sample)
