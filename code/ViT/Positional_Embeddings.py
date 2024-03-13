# Importing PyTorch-related Libraries.
import torch
import torch.nn as nn


# Importing General Libraries.
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



def get_positional_embeddings_SPE(sequence_length, d):
    """
    Generate Positional Embeddings for the Transformer Model.

    Parameters:
    - sequence_length (int): Length of the input sequence.
    - d (int): Dimension of the embeddings.

    Returns:
    torch.Tensor: Positional Embeddings tensor of shape (sequence_length, d).
    """
    # Generate a tensor of positions from 0 to sequence_length - 1.
    positions = torch.arange(0, sequence_length).float().view(-1, 1).to("cuda")

    # Calculate div_term for both sin and cos terms.
    div_term = torch.exp(torch.arange(0, d, 2).float() * -(np.log(10000.0) / d)).to("cuda")

    # Initialize the embeddings tensor with zeros.
    embeddings = torch.zeros(sequence_length, d).to("cuda")

    # Compute sin and cos terms and assign them to the embeddings tensor.
    embeddings[:, 0::2] = torch.sin(positions / div_term).to("cuda")
    embeddings[:, 1::2] = torch.cos(positions / div_term).to("cuda")

    return embeddings



def get_positional_embeddings_RoPE(sequence_length, d):
    """
    Generate Rotary Positional Embeddings for the Transformer Model.

    Parameters:
    - sequence_length (int): Length of the input sequence.
    - d (int): Dimension of the embeddings.

    Returns:
    torch.Tensor: Rotary Positional Embeddings tensor of shape (sequence_length, d).
    """
    # Generate a tensor of positions from 0 to sequence_length - 1.
    positions = torch.arange(0, sequence_length).float().view(-1, 1).to("cuda")

    # Compute sin and cos terms directly using powers of 2.
    embeddings = torch.zeros(sequence_length, d).to("cuda")
    embeddings[:, 0::2] = torch.sin(positions / 2 ** (torch.arange(0, d, 2).float().to("cuda") / d)).to("cuda")
    embeddings[:, 1::2] = torch.cos(positions / 2 ** (torch.arange(1, d, 2).float().to("cuda") / d)).to("cuda")

    return embeddings



def get_positional_embeddings_BERT(sequence_length, hidden_dim):
    """
    Generate Positional Embeddings similar to BERT.

    Parameters:
    - sequence_length (int): Length of the input sequence.
    - hidden_dim (int): Dimension of the embeddings.

    Returns:
    torch.Tensor: Positional Embeddings tensor of shape (1, sequence_length, hidden_dim).
    """
    # Generate a tensor of positions from 0 to sequence_length - 1.
    positions = torch.arange(0, sequence_length).float().view(-1, 1).to("cuda")

    # Calculate div_term for both sin and cos terms.
    div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * -(torch.log(torch.tensor(10000.0)) / hidden_dim)).to("cuda")

    # Initialize the embeddings tensor with zeros.
    embeddings = torch.zeros(sequence_length, hidden_dim).to("cuda")

    # Compute sin and cos terms and assign them to the embeddings tensor.
    embeddings[:, 0::2] = torch.sin(positions / div_term)
    embeddings[:, 1::2] = torch.cos(positions / div_term)

    # Reshape embeddings to match BERT's positional embeddings shape (1, sequence_length, hidden_dim)
    embeddings = embeddings.unsqueeze(0).to("cuda")

    return nn.Parameter(embeddings, requires_grad=False)



# Helper function to Visualize Positional Embeddings.
def visualize_positional_embeddings(embeddings, type_emb):
    """
    Visualize the Positional Embeddings.

    Parameters:
    - embeddings (torch.Tensor): Positional embeddings tensor.

    Returns:
    None
    """

    # Get the number of dimensions (d) from the Embeddings Tensor.
    d = embeddings.size(1)

    # Set the figure size for a larger image.
    plt.figure(figsize=(12, 6))

    # Plot each dimension separately.
    for i in range(d):
        plt.plot(embeddings[:, i].cpu().numpy(), label=f'Dimension {i}')

    # Set plot labels.
    plt.xlabel('Position')
    plt.ylabel('Embedding Value')
    plt.title(f'{type_emb}: Visualization of Positional Embeddings')

    # Place the legend on the right and diminish its size.
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')

    # Show the plot.
    plt.show()



# Helper function to Visualize Positional Embeddings as a Heatmap.
def visualize_positional_embeddings_heatmap(embeddings, type_emb):
    """
    Visualize the Positional Embeddings as a Heatmap.

    Parameters:
    - embeddings (torch.Tensor): Positional embeddings tensor.

    Returns:
    None
    """

    # Get the number of dimensions (d) from the Embeddings Tensor.
    d = embeddings.size(1)

    # Set the figure size for a larger image.
    plt.figure(figsize=(12, 6))

    # Create a heatmap for the positional embeddings.
    sns.heatmap(embeddings.T.cpu().numpy(), cmap='viridis', cbar_kws={'label': 'Embedding Value'})

    # Set plot labels and title.
    plt.xlabel('Position')
    plt.ylabel('Dimension')
    plt.title(f'{type_emb}: Visualization of Positional Embeddings (Heatmap)')

    # Show the plot.
    plt.show()
    





'''

# Run the code to visualize the Positional Embeddings.
if __name__ == "__main__":
    
    # Define parameters
    sequence_length = 100
    d = 64

    # Get positional embeddings using different methods
    embeddings_SPE = get_positional_embeddings_SPE(sequence_length, d)
    embeddings_RoPE = get_positional_embeddings_RoPE(sequence_length, d)
    embeddings_BERT = get_positional_embeddings_BERT(sequence_length, d)

    # Visualize positional embeddings
    visualize_positional_embeddings(embeddings_SPE, "Standard Positional Embeddings")
    visualize_positional_embeddings(embeddings_RoPE, "Rotary Positional Embeddings")
    visualize_positional_embeddings_heatmap(embeddings_BERT.squeeze(0), "BERT Positional Embeddings")

'''