# Importing PyTorch-related Libraries.
import torch
import torch.nn as nn

import math
import torch.nn.functional as F



def attention_MHSA_Cosine(q, k, v, mask=None):
    """
    Compute scaled cosine attention.

     Args:
     q: Queries tensor of shape (batch_size, num_heads, seq_length, head_dim).
     k: Keys tensor of shape (batch_size, num_heads, seq_length, head_dim).
     v: Values tensor of shape (batch_size, num_heads, seq_length, head_dim).
     mask: Optional mask tensor for masked attention.

    Returns:
     values: Tensor of shape (batch_size, seq_length, num_heads, head_dim).
     attention: Tensor of shape (batch_size, num_heads, seq_length, seq_length), containing attention weights.
   """
    # Calculate cosine similarity (skip normalizing key vectors)
    q_norm = F.normalize(q, p=2, dim=-1)
    cosine_sim = torch.matmul(q_norm, k.transpose(-2, -1))
  
    # Apply mask if provided
    if mask is not None:
      cosine_sim = cosine_sim.masked_fill(mask == 0, -1e9)

    # Convert cosine similarity to attention weights using softmax
    attention = F.softmax(cosine_sim, dim=-1)

    # Weighted sum of values
    values = torch.matmul(attention, v)

    return values, attention




def attention_MHSA_Dot_Product(q, k, v, mask=None):
    """
    Compute scaled dot-product attention.

    Args:
        q: Queries tensor of shape (batch_size, num_heads, seq_length, head_dim).
        k: Keys tensor of shape (batch_size, num_heads, seq_length, head_dim).
        v: Values tensor of shape (batch_size, num_heads, seq_length, head_dim).
        mask: Optional mask tensor for masked attention.

    Returns:
        values: Tensor of shape (batch_size, seq_length, num_heads, head_dim).
        attention: Tensor of shape (batch_size, num_heads, seq_length, seq_length), containing attention weights.
    """
    # Calculate attention logits
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    
    # Apply mask if provided
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    
    # Compute attention weights
    attention = F.softmax(attn_logits, dim=-1)
    
    # Weighted sum of values
    values = torch.matmul(attention, v)
    
    return values, attention




# Helper function to support different mask shapes.
# Output shape supports (batch_size, number of heads, seq length, seq length)
# If 2D: broadcasted over batch size and number of heads
# If 3D: broadcasted over number of heads
# If 4D: leave as is
def expand_mask(mask):
    """
    Expand the mask tensor to support different mask shapes.

    Args:
        mask (torch.Tensor): Mask tensor of shape (batch_size, seq_length, seq_length).

    Returns:
        torch.Tensor: Expanded mask tensor of shape (batch_size, num_heads, seq_length, seq_length).
    """
    assert mask.ndim >= 2, "Mask must be at least 2-dimensional with seq_length x seq_length"
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    while mask.ndim < 4:
        mask = mask.unsqueeze(0)
    return mask


class MHSA(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        """
        Initialize the Multi-Head Self-Attention (MHSA) module.

        Args:
            input_dim (int): Dimension of the input.
            embed_dim (int): Dimension of the embeddings.
            num_heads (int): Number of attention heads.
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None, return_attention=False):
        """
        Forward pass of the Multi-Head Self-Attention (MHSA) module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_dim).
            mask (torch.Tensor, optional): Mask tensor for masked attention. Default is None.
            return_attention (bool, optional): Whether to return attention weights. Default is False.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, embed_dim).
            torch.Tensor: Attention weights tensor of shape (batch_size, num_heads, seq_length, seq_length).
        """
        batch_size, seq_length, _ = x.size()
        if mask is not None:
            mask = expand_mask(mask)
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output.
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs.
        values, attention = attention_MHSA_Dot_Product(q, k, v, mask=mask)
        # values, attention = attention_MHSA_Cosine(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o


