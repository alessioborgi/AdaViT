# Importing PyTorch-related Libraries.
import torch
import torch.nn as nn

def attention_MHSA_Dot_Product(q, k, v, d_head):
    """
    Multi-Head Self Attention (MHSA) mechanism with Dot Product.

    Parameters:
    - q (torch.Tensor): Query tensor with shape (seq_length, d_head).
    - k (torch.Tensor): Key tensor with shape (seq_length, d_head).
    - v (torch.Tensor): Value tensor with shape (seq_length, d_head).
    - d_head (int): Dimension of each attention head.

    Returns:
    torch.Tensor: Resultant attention tensor after applying Softmax.
    """

    # Calculate attention scores using the scaled dot-product attention formula.
    attention_scores = q @ k.T / (d_head ** 0.5)

    # Apply Softmax activation along the last dimension to obtain attention weights.
    attention_weights = nn.Softmax(dim=-1)(attention_scores)

    # Multiply attention weights by the value tensor to obtain the attended values.
    attended_values = attention_weights @ v

    return attended_values



def generalized_attention_MHSA_Cosine(q, k, v, d_head):
    """
    Multi-Head Self Attention (MHSA) mechanism with Generalized Attention using Cosine Similarity.

    Parameters:
    - q (torch.Tensor): Query tensor with shape (seq_length, d_head).
    - k (torch.Tensor): Key tensor with shape (seq_length, d_head).
    - v (torch.Tensor): Value tensor with shape (seq_length, d_head).
    - d_head (int): Dimension of each attention head.

    Returns:
    torch.Tensor: Resultant attention tensor after applying Generalized Attention with Cosine Similarity.
    """

    # Scale the query and key vectors.
    q_scaled = q / (d_head ** 0.5)
    k_scaled = k / (d_head ** 0.5)

    # Calculate attention scores using cosine similarity.
    attention_scores = nn.CosineSimilarity(dim=2, eps=1e-6)(q_scaled.unsqueeze(0), k_scaled)

    # Apply softmax activation along the last dimension to obtain attention weights.
    attention_weights = nn.Softmax(dim=-1)(attention_scores)

    # Multiply attention weights by the value tensor to obtain the attended values.
    attended_values = attention_weights @ v

    return attended_values




class MHSA(nn.Module):

    def __init__(self, d, n_heads=2):
        """
        Multi-Head Self Attention (MHSA) Module.

        Parameters:
        - d (int): Dimension of the input tokens.
        - n_heads (int): Number of attention heads.

        Returns:
        None
        """

        super(MHSA, self).__init__()
        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"

        # Split the dimension into n_heads parts.
        d_head = int(d / n_heads)

        # Linear mappings for Query(q), Key(k), and Value(v) for each head.
        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])

        # Number of Heads.
        self.d_head = d_head

        # Trainable Output Matrix.
        # self.W_output = nn.Linear(n_heads * d_head, d)

        # Softmax Definition.
        self.softmax = nn.Softmax(dim=-1)

        # Initialize weights.
        self.initialize_weights_msa()

    def forward(self, sequences):
        """
        Forward pass of the MHSA module.

        Parameters:
        - sequences (torch.Tensor): Input token sequences with shape (N, seq_length, token_dim).

        Returns:
        torch.Tensor: Output tensor after MHSA with shape (N, seq_length, item_dim).
        """

        result = []
        for sequence in sequences:

            seq_result = []
            for head in range(self.n_heads):

                # Compute the q,k,v for every head.
                q_mapping, k_mapping, v_mapping = self.q_mappings[head], self.k_mappings[head], self.v_mappings[head]

                # Extract the corresponding part of the sequence for the current head.
                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                # Calculate Attention Scores with one of the methods.
                # 1) Basic MHSA with Dot Product.
                attention = attention_MHSA_Dot_Product(q, k, v, self.d_head)

                # 2) Generalized MHSA with Cosine Similarity.
                #attention = generalized_attention_MHSA_Cosine(q, k, v, self.d_head)

                # Append the Attention Scores.
                seq_result.append(attention)

            # Apply the trainable output matrix W_output.
            # seq_result = self.W_output(seq_result)

            # Concatenate the results coming from the different Heads and Stack Vertically the result.
            result.append(torch.hstack(seq_result).to("cuda"))

        # Concatenate results for all the sequences.
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result]).to("cuda")


    def initialize_weights_msa(self):
        """
        Initialize weights for linear layers in the MHSA module.

        Parameters:
        None

        Returns:
        None
        """

        # Initialize weights for the q, k, v values.
        for q_mapping, k_mapping, v_mapping in zip(self.q_mappings, self.k_mappings, self.v_mappings):
            nn.init.xavier_uniform_(q_mapping.weight)
            nn.init.xavier_uniform_(k_mapping.weight)
            nn.init.xavier_uniform_(v_mapping.weight)

        # Initialize weights for the output matrix W_output.
        # nn.init.xavier_uniform_(self.W_output.weight)


