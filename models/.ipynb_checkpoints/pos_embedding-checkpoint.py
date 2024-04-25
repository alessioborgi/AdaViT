import torch
import torch.nn as nn
import math

class SinusoidalPositionalEmbedding(nn.Module):
    """
    Sinusoidal Positional Embedding module for Transformer models.

    Args:
        seq_length (int): Length of the input sequence.
        hidden_dim (int): Dimension of the embeddings.

    Attributes:
        seq_length (int): Length of the input sequence.
        hidden_dim (int): Dimension of the embeddings.
    """

    def __init__(self, seq_length, hidden_dim):
        super(SinusoidalPositionalEmbedding, self).__init__()
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim

    def forward(self, x):
        """
        Forward pass of the Sinusoidal Positional Embedding module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Positional embeddings tensor of shape (seq_length, hidden_dim).
        """
        position = torch.arange(0, self.seq_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.hidden_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / self.hidden_dim))
        position_encoding = torch.zeros(1, self.seq_length, self.hidden_dim, dtype=torch.float32)
        position_encoding[0, :, 0::2] = torch.sin(position * div_term)
        position_encoding[0, :, 1::2] = torch.cos(position * div_term)
        return position_encoding
    
    

class BERTPositionalEmbedding(nn.Module):
    """
    BERT Positional Embedding module for Transformer models.

    Args:
        seq_length (int): Length of the input sequence.
        hidden_dim (int): Dimension of the embeddings.

    Attributes:
        pos_embedding (nn.Parameter): Learnable positional embeddings parameter.
    """

    def __init__(self, seq_length, hidden_dim):
        super(BERTPositionalEmbedding, self).__init__()
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))

    def forward(self, x):
        """
        Forward pass of the BERT Positional Embedding module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Positional embeddings tensor of shape (seq_length, hidden_dim).
        """
        return self.pos_embedding
    
    

class RelativePositionalEmbedding(nn.Module):
    """
    Relative Positional Embedding module for Transformer models.

    Args:
        seq_length (int): Length of the input sequence.
        hidden_dim (int): Dimension of the embeddings.
        num_buckets (int): Number of buckets for relative positions.

    Attributes:
        seq_length (int): Length of the input sequence.
        hidden_dim (int): Dimension of the embeddings.
        num_buckets (int): Number of buckets for relative positions.
        embedding (nn.Parameter): Learnable embedding parameter.
    """

    def __init__(self, seq_length, hidden_dim, num_buckets):
        super(RelativePositionalEmbedding, self).__init__()
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.num_buckets = num_buckets
        self.embedding = nn.Parameter(torch.randn(num_buckets, hidden_dim))

    def forward(self, x):
        """
        Forward pass of the Relative Positional Embedding module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Relative positional embeddings tensor of shape (seq_length, hidden_dim).
        """
        # Placeholder implementation for demonstration purposes
        return torch.zeros(self.seq_length, self.hidden_dim)
