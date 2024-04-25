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
        position = torch.arange(0, self.seq_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.hidden_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / self.hidden_dim))
        position_encoding = torch.zeros(1, self.seq_length, self.hidden_dim, dtype=torch.float32)
        position_encoding[0, :, 0::2] = torch.sin(position * div_term)
        position_encoding[0, :, 1::2] = torch.cos(position * div_term)
        self.pos_embedding = nn.Parameter(position_encoding)

    def forward(self):
        """
        Forward pass of the Sinusoidal Positional Embedding module.

        Returns:
            torch.Tensor: Positional embeddings tensor of shape (seq_length, hidden_dim).
        """
        
        return self.pos_embedding

    
    

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
    
    
'''
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
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, hidden_dim).

        Returns:
            torch.Tensor: Output tensor with relative positional embeddings incorporated.
        """
        batch_size, seq_length, hidden_dim = x.size()
        
        # Compute differences in positions
        positions = torch.arange(seq_length, dtype=torch.float32, device=x.device).unsqueeze(1)
        rel_positions = positions - positions.transpose(0, 1)

        # Clip relative positions to ensure they fall within the range [-num_buckets, num_buckets]
        rel_positions_clipped = torch.clamp(rel_positions, -self.num_buckets, self.num_buckets)

        # Map relative positions to embedding indices
        embedding_indices = rel_positions_clipped + self.num_buckets

        # Retrieve embeddings from the embedding parameter
        rel_pos_embeddings = self.embedding[embedding_indices]

        # Reshape relative positional embeddings to match the input tensor shape
        rel_pos_embeddings = rel_pos_embeddings.view(1, seq_length, hidden_dim).expand(batch_size, -1, -1)

        # Add relative positional embeddings to the input tensor
        output = x + rel_pos_embeddings

        return output
        
'''

'''
class RelativePositionalEmbedding(nn.Module):
    def __init__(self, num_patches, hidden_dim, num_buckets=32):
        super().__init__()
        self.num_patches = num_patches
        self.token_embeddings = nn.Embedding(num_buckets * 2 + 1, hidden_dim)
        self.rel_height = nn.Embedding(num_buckets + 1, hidden_dim // 2)
        self.rel_width = nn.Embedding(num_buckets + 1, hidden_dim // 2)
        self.num_buckets = num_buckets

        # Initialize the positional embeddings
        self.initialize_positional_embeddings()

    def initialize_positional_embeddings(self):
        position = torch.arange(0, self.num_patches, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.num_buckets, 2, dtype=torch.float32) * (-math.log(10000.0) / self.num_buckets))
        position_encoding = torch.zeros(1, self.num_patches, self.num_buckets * 2, dtype=torch.float32)
        position_encoding[0, :, 0::2] = torch.sin(position * div_term)
        position_encoding[0, :, 1::2] = torch.cos(position * div_term)
        self.pos_embedding = nn.Parameter(position_encoding)

    def forward(self):
        """
        Forward pass of the Relative Positional Embedding module.

        Returns:
            torch.Tensor: Positional embeddings tensor of shape (num_patches, hidden_dim).
        """
        return self.pos_embedding

'''

class RelativePositionalEmbedding(nn.Module):
    def __init__(self, num_patches, hidden_dim, num_buckets):
        super().__init__()
        self.num_patches = num_patches
        self.num_buckets = num_buckets
        self.token_embeddings = nn.Parameter(num_buckets * 2 + 1, hidden_dim)
        self.rel_height = nn.Embedding(num_buckets + 1, hidden_dim // 2)
        self.rel_width = nn.Embedding(num_buckets + 1, hidden_dim // 2)

    def forward(self):
        device = next(self.parameters()).device
        i, j = torch.meshgrid(torch.arange(self.num_patches), torch.arange(self.num_patches))
        rel_dist = torch.abs(i - j)  # Calculate absolute relative distances

        # Encode relative height and width
        rel_height_embeddings = self.rel_height(torch.clamp(rel_dist, 0, self.num_buckets))
        rel_width_embeddings = self.rel_width(torch.clamp(rel_dist, 0, self.num_buckets))
        rel_embeddings = torch.cat((rel_height_embeddings, rel_width_embeddings), dim=-1)

        # Encode relative position using buckets
        rel_pos_bucket = torch.min(torch.floor(rel_dist / 2), torch.tensor(self.num_buckets - 1))
        bucket_embeddings = self.token_embeddings(rel_pos_bucket.long())

        # Combine embeddings for final relative positional encoding
        return rel_embeddings + bucket_embeddings
