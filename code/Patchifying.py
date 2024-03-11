# Importing PyTorch-related Libraries.
import torch


# Importing General Libraries.
import matplotlib.pyplot as plt

# Importing from other files.
from ViTMain import AViT_DataModule


def Make_Patches_from_Image(images, n_patches):
    """
    Extract patches from input images.

    Parameters:
    - images (torch.Tensor): Input images tensor with shape (batch_size, channels, height, width).
    - n_patches (int): Number of patches in each dimension.

    Returns:
    torch.Tensor: Extracted patches tensor with shape (batch_size, n_patches^2, patch_size^2 * channels).
    """
    # Get the dimensions of the input images.
    n, c, h, w = images.shape

    # Ensure that the input images are square.
    assert h == w, "make_patches_from_image method is implemented for square images only!"

    # Initialize a tensor to store the extracted patches.
    patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2).to("cuda")
    patch_size = h // n_patches

    # Loop over each image in the batch.
    for idx, image in enumerate(images):
        # Loop over each patch in both dimensions.
        for i in range(n_patches):
            for j in range(n_patches):
                # Extract the patch from the image.
                patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                # Flatten the patch and store it in the patches tensor.
                patches[idx, i * n_patches + j] = patch.flatten()

    return patches




# Helper function to Visualize Patches.
def visualize_patches(images, n_patches, title):
    """
    Visualize patches extracted from Images.

    Parameters:
    - images (torch.Tensor): Input images tensor with shape (batch_size, channels, height, width).
    - n_patches (int): Number of patches in each dimension.
    - title (str): Title for the entire subplot.

    Returns:
    None
    """
    # Extract patches from the input images using the make_patches_from_image function.
    patches = Make_Patches_from_Image(images, n_patches)

    # Create a subplot for visualizing patches.
    fig, axs = plt.subplots(n_patches, n_patches, figsize=(8, 8))
    fig.suptitle(title)

    # Calculate the patch size based on the input images.
    patch_size = images.shape[-1] // n_patches

    # Loop over each patch in both dimensions.
    for i in range(n_patches):
        for j in range(n_patches):
            # Calculate the index of the patch.
            patch_index = i * n_patches + j
            # Reshape each patch to (3, patch_size, patch_size).
            patch = patches[0, patch_index].reshape(3, patch_size, patch_size).cpu().numpy()
            # Display the patch in the subplot.
            axs[i, j].imshow(patch.transpose(1, 2, 0))
            axs[i, j].axis('off')

    # Show the entire subplot.
    plt.show()


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

    # Visualize patches for a Training Image.
    visualize_patches(train_batch[0], n_patches=8, title='Training Patches')

    # Visualize patches for a Validation Image.
    visualize_patches(val_batch[0], n_patches=8, title='Validation Patches')

    # Visualize patches for a Test Image.
    visualize_patches(test_batch[0], n_patches=8, title='Test Patches')
