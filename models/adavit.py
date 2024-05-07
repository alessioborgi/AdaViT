import torch
from torch import nn
import torch.nn.functional as F
import math
import random
from typing import Optional, List
from abc import ABC
import os
from torchvision.models.vision_transformer import ViT_B_16_Weights, ViT_B_32_Weights

import numpy as np
from .blocks import SelfAttention, MLP
from .pos_embedding import SinusoidalPositionalEmbedding, BERTPositionalEmbedding
from torch.autograd import Variable
 
"""
Adaptive Vision Transformer (AViT) as per https://arxiv.org/pdf/2112.07658.pdf.
"""



# AViT Block
class AViTBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        gate_scale: float = 10,
        gate_center: float = 30,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.gate_scale = gate_scale
        self.gate_center = gate_center
        
        
        # Attention block
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.self_attention = SelfAttention(hidden_dim, num_heads, attention_dropout)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim=hidden_dim, mlp_dim=mlp_dim)


    
    def forward_act(self, x, mask=None):

        debug=False
        analyze_delta = True
        bs, token, dim = x.shape

        # x is bs, seq_len, token_dim
        # mask is bs, seq_len

        if mask is None:
            x = x + self.self_attention(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
        else:
            x = x + self.self_attention(self.ln_1(x*(1-mask).view(bs, token, 1))*(1-mask).view(bs, token, 1)) #, mask=mask)
            x = x + self.mlp(self.ln_2(x*(1-mask).view(bs, token, 1))*(1-mask).view(bs, token, 1))

        
        gate_scale, gate_center = self.gate_scale, self.gate_center
        halting_score_token = torch.sigmoid(x[:,:,0] * gate_scale - gate_center)
        # initially first position used for layer halting, second for token
        # now discarding position 1
        halting_score = [-1, halting_score_token]
        

        return x, halting_score

    


def speed_up_halting(mask_token, new_halted_tokens_per_layer, percentage, discard_level, patch_width):
    # Set seed for reproducibility
    torch.manual_seed(31)
    
    # Find the indices of True values
    true_indices = torch.nonzero(new_halted_tokens_per_layer, as_tuple=False)

    # Calculate the number of True values to retain based on the percentage
    num_true_to_retain = int(percentage * len(true_indices))

    # Randomly select a subset of indices
    selected_indices = random.sample(range(len(true_indices)), num_true_to_retain)

    # Reset the new_halted_tokens_per_layer to all False
    new_halted_tokens_per_layer.fill_(False)

    # Set the selected indices to True
    for idx in selected_indices:
        new_halted_tokens_per_layer[true_indices[idx][0], true_indices[idx][1]] = True
    
    # 1 --> 1
    if discard_level == "identity":
        # Return the mask token unchanged
        return mask_token

    # 1 --> 3
    elif discard_level == "nearby":
        # Halt the left and right tokens to the mask_token's token that correspond to be a True in the new_halted_tokens_per_layer.
        left_indices = torch.clamp(true_indices - torch.tensor([[0, 1]]).to("cuda"), min=0)
        right_indices = torch.clamp(true_indices + torch.tensor([[0, 1]]).to("cuda"), max=mask_token.size(1) - 1)

        # Check if the indices are not on the border of the image (every patch_width pixels)
        left_border_check = left_indices[:, 1] % patch_width != 0
        right_border_check = (right_indices[:, 1] + 1) % patch_width != 0

        # Apply halting only if the indices are not on the border of the image
        mask_token[true_indices[left_border_check][:, 0], left_indices[left_border_check][:, 1]] = False
        mask_token[true_indices[right_border_check][:, 0], right_indices[right_border_check][:, 1]] = False
    
    # 1 --> 5
    elif discard_level == "cross":
        # Halt the left, right, up, and down tokens to the mask_token's token that correspond to be True in the new_halted_tokens_per_layer.
        left_indices = torch.clamp(true_indices - torch.tensor([[0, 1]]).to("cuda"), min=0)
        right_indices = torch.clamp(true_indices + torch.tensor([[0, 1]]).to("cuda"), max=mask_token.size(1) - 1)
        up_indices = torch.clamp(true_indices - torch.tensor([[1, 0]]).to("cuda"), min=0)
        down_indices = torch.clamp(true_indices + torch.tensor([[1, 0]]).to("cuda"), max=mask_token.size(0) - 1)

        # Check if the indices are not on the border of the image (every patch_width pixels)
        left_border_check = left_indices[:, 1] % patch_width != 0
        right_border_check = (right_indices[:, 1] + 1) % patch_width != 0
        up_border_check = up_indices[:, 0] % patch_width != 0
        down_border_check = (down_indices[:, 0] + 1) % patch_width != 0

        # Apply halting only if the indices are not on the border of the image
        mask_token[true_indices[left_border_check][:, 0], left_indices[left_border_check][:, 1]] = False
        mask_token[true_indices[right_border_check][:, 0], right_indices[right_border_check][:, 1]] = False
        mask_token[true_indices[up_border_check][:, 0], up_indices[up_border_check][:, 1]] = False
        mask_token[true_indices[down_border_check][:, 0], down_indices[down_border_check][:, 1]] = False
    
    # 1 --> 9
    # 13 halted tokens are: The True token itself (1 token)
    # Two tokens to the right
    # Two tokens to the left
    # Two tokens above
    # Two tokens below
    # Four tokens diagonally (one in each direction: top-right, top-left, bottom-right, bottom-left)
    elif discard_level == "square":
        # Halt the left, right, up, down, top-left, top-right, bottom-left, and bottom-right tokens to the mask_token's token 
        # that correspond to be True in the new_halted_tokens_per_layer.
        left_indices = torch.clamp(true_indices - torch.tensor([[0, 1]]).to("cuda"), min=0)
        right_indices = torch.clamp(true_indices + torch.tensor([[0, 1]]).to("cuda"), max=mask_token.size(1) - 1)
        up_indices = torch.clamp(true_indices - torch.tensor([[1, 0]]).to("cuda"), min=0)
        down_indices = torch.clamp(true_indices + torch.tensor([[1, 0]]).to("cuda"), max=mask_token.size(0) - 1)
        top_left_indices = torch.clamp(true_indices - torch.tensor([[1, 1]]).to("cuda"), min=0)
        top_right_indices = torch.clamp(true_indices - torch.tensor([[1, -1]]).to("cuda"), min=0)
        bottom_left_indices = torch.clamp(true_indices + torch.tensor([[1, -1]]).to("cuda"), max=mask_token.size(0) - 1)
        bottom_right_indices_row = torch.clamp(true_indices[:, 0] + 1, max=mask_token.size(0) - 1)
        bottom_right_indices_col = torch.clamp(true_indices[:, 1] + 1, max=mask_token.size(1) - 1)
        bottom_right_indices = torch.stack((bottom_right_indices_row, bottom_right_indices_col), dim=1)

        # Check if the indices are not on the border of the image (every patch_width pixels).
        left_border_check = left_indices[:, 1] % patch_width != 0
        right_border_check = (right_indices[:, 1] + 1) % patch_width != 0
        up_border_check = up_indices[:, 0] % patch_width != 0
        down_border_check = (down_indices[:, 0] + 1) % patch_width != 0
        top_left_border_check = (top_left_indices[:, 0] % patch_width != 0) & (top_left_indices[:, 1] % patch_width != 0)
        top_right_border_check = (top_right_indices[:, 0] % patch_width != 0) & ((top_right_indices[:, 1] + 1) % patch_width != 0)
        bottom_left_border_check = ((bottom_left_indices[:, 0] + 1) % patch_width != 0) & (bottom_left_indices[:, 1] % patch_width != 0)
        bottom_right_border_check = ((bottom_right_indices[:, 0] + 1) % patch_width != 0) & ((bottom_right_indices[:, 1] + 1) % patch_width != 0)

        # Halt the tokens at the True indices
        mask_token[true_indices[:, 0], true_indices[:, 1]] = False

        # Apply halting to tokens at the defined indices
        mask_token[left_indices[left_border_check][:, 0], left_indices[left_border_check][:, 1]] = False
        mask_token[right_indices[right_border_check][:, 0], right_indices[right_border_check][:, 1]] = False
        mask_token[up_indices[up_border_check][:, 0], up_indices[up_border_check][:, 1]] = False
        mask_token[down_indices[down_border_check][:, 0], down_indices[down_border_check][:, 1]] = False
        mask_token[top_left_indices[top_left_border_check][:, 0], top_left_indices[top_left_border_check][:, 1]] = False
        mask_token[top_right_indices[top_right_border_check][:, 0], top_right_indices[top_right_border_check][:, 1]] = False
        mask_token[bottom_left_indices[bottom_left_border_check][:, 0], bottom_left_indices[bottom_left_border_check][:, 1]] = False
        mask_token[bottom_right_indices[bottom_right_border_check][:, 0], bottom_right_indices[bottom_right_border_check][:, 1]] = False

        # Additional halting for diagonal tokens
        mask_token[true_indices[top_left_indices[:, 0], top_left_indices[:, 1]]] = False
        mask_token[true_indices[top_right_indices[:, 0], top_right_indices[:, 1]]] = False
        mask_token[true_indices[bottom_left_indices[:, 0], bottom_left_indices[:, 1]]] = False
        mask_token[true_indices[bottom_right_indices[:, 0], bottom_right_indices[:, 1]]] = False


        
    # 1 --> 13
    elif discard_level == "isotropic":
        # Define indices for halting tokens in all directions
        left_indices = torch.clamp(true_indices - torch.tensor([[0, 1]]).to("cuda"), min=0)
        right_indices = torch.clamp(true_indices + torch.tensor([[0, 1]]).to("cuda"), max=mask_token.size(1) - 1)
        up_indices = torch.clamp(true_indices - torch.tensor([[1, 0]]).to("cuda"), min=0)
        down_indices = torch.clamp(true_indices + torch.tensor([[1, 0]]).to("cuda"), max=mask_token.size(0) - 1)
        top_left_indices = torch.clamp(true_indices - torch.tensor([[1, 1]]).to("cuda"), min=0)
        top_right_indices = torch.clamp(true_indices - torch.tensor([[1, -1]]).to("cuda"), min=0)
        bottom_left_indices = torch.clamp(true_indices + torch.tensor([[1, -1]]).to("cuda"), max=mask_token.size(0) - 1)
        bottom_right_indices_row = torch.clamp(true_indices[:, 0] + 1, max=mask_token.size(0) - 1)
        bottom_right_indices_col = torch.clamp(true_indices[:, 1] + 1, max=mask_token.size(1) - 1)
        bottom_right_indices = torch.stack((bottom_right_indices_row, bottom_right_indices_col), dim=1)

        # Check if the indices are not on the border of the image (every patch_width pixels).
        left_border_check = left_indices[:, 1] % patch_width != 0
        right_border_check = (right_indices[:, 1] + 1) % patch_width != 0
        up_border_check = up_indices[:, 0] % patch_width != 0
        down_border_check = (down_indices[:, 0] + 1) % patch_width != 0
        top_left_border_check = (top_left_indices[:, 0] % patch_width != 0) & (top_left_indices[:, 1] % patch_width != 0)
        top_right_border_check = (top_right_indices[:, 0] % patch_width != 0) & ((top_right_indices[:, 1] + 1) % patch_width != 0)
        bottom_left_border_check = ((bottom_left_indices[:, 0] + 1) % patch_width != 0) & (bottom_left_indices[:, 1] % patch_width != 0)
        bottom_right_border_check = ((bottom_right_indices[:, 0] + 1) % patch_width != 0) & ((bottom_right_indices[:, 1] + 1) % patch_width != 0)

        # Halt the tokens at the True indices
        mask_token[true_indices[:, 0], true_indices[:, 1]] = False

        # Apply halting to tokens at the defined indices if they are within the image boundaries
        mask_token[left_indices[left_border_check][:, 0], left_indices[left_border_check][:, 1]] = False
        mask_token[right_indices[right_border_check][:, 0], right_indices[right_border_check][:, 1]] = False
        mask_token[up_indices[up_border_check][:, 0], up_indices[up_border_check][:, 1]] = False
        mask_token[down_indices[down_border_check][:, 0], down_indices[down_border_check][:, 1]] = False
        mask_token[top_left_indices[top_left_border_check][:, 0], top_left_indices[top_left_border_check][:, 1]] = False
        mask_token[top_right_indices[top_right_border_check][:, 0], top_right_indices[top_right_border_check][:, 1]] = False
        mask_token[bottom_left_indices[bottom_left_border_check][:, 0], bottom_left_indices[bottom_left_border_check][:, 1]] = False
        mask_token[bottom_right_indices[bottom_right_border_check][:, 0], bottom_right_indices[bottom_right_border_check][:, 1]] = False

        # Additional halting for diagonal tokens within the image boundaries
        mask_token[true_indices[top_left_indices[:, 0], top_left_indices[:, 1]]] = False
        mask_token[true_indices[top_right_indices[:, 0], top_right_indices[:, 1]]] = False
        mask_token[true_indices[bottom_left_indices[:, 0], bottom_left_indices[:, 1]]] = False
        mask_token[true_indices[bottom_right_indices[:, 0], bottom_right_indices[:, 1]]] = False


    else: 
        print("The discard level doesn't reflect any of the listed discard level. It should be one in {identity, nearby, cross, square, isotropic}")

    return mask_token


    


# ViT Encoder
class AViTEncoder(nn.Module):

    def __init__(
        self,
        seq_length: int,
        patch_width: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        discard_level: str,
        percentage: float,
        eps: float = 0.01,
        gate_scale: float = 10,
        gate_center: float = 30,
    ):
        super().__init__()

        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.eps = eps
        self.patch_width = patch_width
        
        ############### Positional Embedding ###############
        # 1) BERT
        self.pos_embedding = BERTPositionalEmbedding(seq_length, hidden_dim).pos_embedding
        
        # 2) Sinusoidal Positional Embedding (SPE)
        #self.pos_embedding = SinusoidalPositionalEmbedding(seq_length, hidden_dim).pos_embedding
        
        # 3) Relative Positional Embedding (RPE)
        #num_buckets = 16
        #self.pos_embedding = RelativePositionalEmbedding(seq_length, hidden_dim, num_buckets)
        
        ############################################################
        
        self.dropout = nn.Dropout(dropout)
        layers: List = []
        for i in range(num_layers):
            layers.append(AViTBlock(
                            num_heads,
                            hidden_dim,
                            mlp_dim,
                            dropout,
                            attention_dropout,
                            gate_scale,
                            gate_center
                            ))

        self.layers = nn.ModuleList(layers)
        self.ln = nn.LayerNorm(hidden_dim)
        
        # Instantiating the structure to get statistics over the Halting Metrics.
        self.num_halted_tokens_per_layer = [0 for _ in range(num_layers)]# for each layer we have the sum of halted tokens

        # for token act part
        self.c_token = None
        self.R_token = None
        self.mask_token = None
        self.rho_token = None
        self.counter_token = None
        self.seq_length = seq_length

        


    def forward(self, input: torch.Tensor):
        
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        
        # Shape before pos_embedding is [128, 197, 192]
        input = input + self.pos_embedding

        # Shape after pos_embedding is the same, i.e.: [128, 197, 192]. Indeed it's an addition not a concatenation!
        input = self.dropout(input)
                
        return self.forward_features_act_token(input)


    def forward_features_act_token(self, x):
        
        # Now we start the action part. x.shape is [128, 197, 192]. 
        bs = x.size()[0]  # The batch size 
        


        if self.c_token is None or bs != self.c_token.size()[0]:
            
            self.c_token = Variable(torch.zeros(bs, self.seq_length).cuda())
            self.R_token = Variable(torch.ones(bs, self.seq_length).cuda())
            self.mask_token = Variable(torch.ones(bs, self.seq_length).cuda())
            self.rho_token = Variable(torch.zeros(bs, self.seq_length).cuda())
            self.counter_token = Variable(torch.ones(bs, self.seq_length).cuda())

        c_token = self.c_token.clone()
        R_token = self.R_token.clone()
        mask_token = self.mask_token.clone()
        self.rho_token = self.rho_token.detach() * 0.
        self.counter_token = self.counter_token.detach() * 0 + 1.
        
        # Will contain the output of this residual layer (weighted sum of outputs of the residual blocks)
        output = None
        # Use out to backbone
        out = x
        
        # Initialize list for seeing the # of halted tokens per layer.
        self.halting_score_layer = []
        # Initialize additional halted tokens mask. This will report the new_halted_tokens w.r.t. the previous layer.
        new_halted_tokens_per_layer = torch.zeros(bs, self.seq_length).bool().cuda()  # Initialize as boolean mask  
        
        # For each of the 12 layers.
        for i, adaptive_layer in enumerate(self.layers):

            # Block out all the parts that are not used
            out.data = out.data * mask_token.float().view(bs, self.seq_length, 1)     # out.data.shape = [128, 197, 192].
            

            # Evaluate layer and get Halting Probability for Each Sample
            # block_output, h_lst = l.forward_act(out)    # h is a vector of length bs, block_output a 3D tensor
            block_output, h_lst = adaptive_layer.forward_act(out, 1.-mask_token.float())    # h is a vector of length bs, block_output a 3D tensor
            # block_output.shape = [128, 197, 192]. Tensor
            # dim(h_lst) = [-1, halting_scores]. List

            self.halting_score_layer.append(torch.mean(h_lst[1][1:]))

            out = block_output.clone()              # Deep copy needed for the next layer
            # out.shape = [128, 197, 192]

            _, h_token = h_lst # h is layer_halting score, h_token is token halting score, first position discarded
            # h_token.shape = [128, 197]

            # here, 1 is remaining, 0 is blocked
            block_output = block_output * mask_token.float().view(bs, self.seq_length, 1)
            # block_output.shape = [128, 197, 192]

            # Is this the last layer in the block?
            if i==len(self.layers)-1:
                h_token = nn.Parameter(torch.ones(bs, self.seq_length).cuda())
            
            # h_token.shape = [128, 197]
            # c_token.shape = [128, 197]
            # For token part
            c_token = c_token + h_token
            self.rho_token = self.rho_token + mask_token.float()

            
            ############### Case 1: Halting Threshold Reached ###############
            # Computing the reached_token mask T/F.
            reached_token = c_token >= 1 - self.eps    # Tensor of boolean values True/False
            
        
            # Counting Number of Halted Tokens at layer i.
            num_halted = torch.sum(reached_token) 
            new_halted_tokens = reached_token & (~mask_token.bool())  # Newly halted tokens
            new_halted_tokens_per_layer |= new_halted_tokens  # Update additional halted tokens mask
            #self.num_halted_tokens_per_layer[i] += num_halted
            
            # Transform the reached Token into float.
            reached_token = reached_token.float() * mask_token.float()         # reached_token.shape = [128, 197]
            # reached_token is now literally a mask containing either 0. or 1. ( float).
            
            # R_token represents the "remaining" halting scores for each token.
            # delta1 contains the contributions of tokens that have reached their halting threshold to the output of the current layer. 
            # These contributions are weighted by the remaining halting scores of each token (R_token). 
            # In other words, delta1 represents the "active" or "contributing" tokens in the output of the layer. 
            delta1 = block_output * R_token.view(bs, self.seq_length, 1) * reached_token.view(bs, self.seq_length, 1)
            
            # self.rho_token contains the cumulative sum of halting scores across layers for each token. 
            # This product represents the contribution of each token that has reached its halting threshold in advancing through the layers. 
            # By summing these contributions over all layers, self.rho_token provides a measure of the overall progress or "importance" of each token in the computation process.
            # In essence, self.rho_token reflects how much each token has been involved in the computation across all layers, with higher values indicating tokens that have played a more significant role in the final output.
            self.rho_token = self.rho_token + R_token * reached_token
            
            
            ############### Case 2: Halting Threshold Not Reached yet ###############
            # token part
            not_reached_token = c_token < 1 - self.eps
            not_reached_token = not_reached_token.float()
            R_token = R_token - (not_reached_token.float() * h_token)
            delta2 = block_output * h_token.view(bs, self.seq_length, 1) * not_reached_token.view(bs, self.seq_length, 1)

            self.counter_token = self.counter_token + not_reached_token # These data points will need at least one more layer

            # Update the Mask.
            # The update is done based on whether their cumulative halting score (c_token) is less than the threshold 1 - self.eps. 
            # If the cumulative halting score of a token is below the threshold, it means that the token has not yet reached its halting threshold and should be allowed to proceed to the next layer (corresponding entry = 1),i.e., 
            #    notmasked. (True)
            # Else, if the cumulative halting score of a token is equal to or greater than the threshold, it means that the token has reached its halting threshold and should be halted, preventing it from further processing in 
            #    subsequent layers (corresponding entry = 0),i.e., Masked. (False)
            mask_token = c_token < 1 - self.eps       # Tensor of True/False values.
            
            # Find the indices of False (Halted) values
            num_zeros = (mask_token == 0).sum().item()
            #print("Number of zeros BEFORE SPEED-UP:", num_zeros) 
            # mask_token.shape = [128, 197]
            # new_halted_tokens_per_layer.shape = [128, 197]  and contains all False or True.
            #################### SPEED-UP HALTING NOVELTY ####################################################################################################
            
            # Call speed_up_halting function to modify the mask_token
            
            #print("The image_size is equal to: ", self.image_size)
            mask_token = speed_up_halting(mask_token, new_halted_tokens_per_layer, percentage=self.percentage, discard_level=self.discard_level, patch_width=self.patch_width)
            ##################################################################################################################################################
                        
            # Find the indices of False (Halted) values
            num_zeros = (mask_token == 0).sum().item()
            #print("Number of zeros AFTER SPEED-UP:", num_zeros)            
            self.num_halted_tokens_per_layer[i] += num_zeros
            
            
            if output is None:
                output = delta1 + delta2
            else:
                output = output + (delta1 + delta2)

        x = self.ln(output)

        return x
            
        
        
        
        

class AdaptiveVisionTransformer(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""


    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        discard_level: str = "identity",
        percentage: float = 1.0,
        num_classes: int = 1000,
        representation_size: Optional[int] = None,
        num_registers: int = 0,
        num_class_tokens: int = 1,
        eps: float = 0.01,
        gate_scale: float = 10,
        gate_center: float = 30,
        torch_pretrained_weights: Optional[str] = None,
        timm_pretrained_weights: Optional[List] = None,
    ):
        """
        Args:
            image_size (int): The size of the input image.
            patch_size (int): The size of each patch in the image.
            num_layers (int): The number of layers in the transformer encoder.
            num_heads (int): The number of attention heads in the transformer encoder.
            hidden_dim (int): The hidden dimension size in the transformer encoder.
            mlp_dim (int): The dimension size of the feed-forward network in the transformer encoder.
            dropout (float, optional): The dropout rate. Defaults to 0.0.
            attention_dropout (float, optional): The dropout rate for attention weights. Defaults to 0.0.
            num_classes (int, optional): The number of output classes. Defaults to 1000.
            representation_size (int, optional): The size of the output representation. Defaults to None.
            num_registers (int, optional): The number of register tokens to be added. Defaults to 0.
            num_class_tokens (int, optional): The number of class tokens to be added. Defaults to 1.
            eps (float, optional): The epsilon value for the ACT. Defaults to 0.01.
            gate_scale (float, optional): The scale value for the ACT. Defaults to 10.
            gate_center (float, optional): The center value for the ACT. Defaults to 30.
            torch_pretrained_weights (str, optional): The path to the pretrained weights in the Torch format. Defaults to None
                Example: 'ViT_B_16_Weights[IMAGENET1K_V1]'.
                See options at https://github.com/pytorch/vision/blob/a52607ece94aedbe41107617ace22a8da91efc25/torchvision/models/vision_transformer.py#L351
            timm_pretrained_weights (List, optional): The path to the pretrained weights in the Timm format. Defaults to None. 
                Example: ['facebookresearch/deit_base_patch16_224', 'deit_base_patch16_224']
        """
        
        super().__init__()
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.discard_level = discard_level
        self.percentage = percentage
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.num_heads = num_heads
        self.num_registers = num_registers
        self.num_class_tokens = num_class_tokens
        self.num_layers = num_layers
        self.eps = eps
        self.gate_scale = gate_scale
        self.gate_center = gate_center
        self.patch_width = self.image_size / self.patch_size
        


        self.conv_proj = nn.Conv2d(in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size)

        seq_length = (image_size // patch_size) ** 2
        # Add class tokens
        self.class_tokens = nn.Parameter(torch.zeros(1, num_class_tokens, hidden_dim))
        seq_length += num_class_tokens

        # Add registers
        if num_registers > 0:
            self.register_tokens = nn.Parameter(torch.zeros(1, num_registers, hidden_dim))
            seq_length += num_registers

        self.encoder = AViTEncoder(
            seq_length,
            self.patch_width,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            self.discard_level,
            self.percentage,
            eps,
            gate_scale,
            gate_center,
            )

        
        self.seq_length = seq_length

        self.head = nn.Linear(hidden_dim, num_classes)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
    

        fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
        nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
        if self.conv_proj.bias is not None:
            nn.init.zeros_(self.conv_proj.bias)
        
        self.load_weights(torch_pretrained_weights, timm_pretrained_weights)


    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Add registers
        if self.num_registers > 0:
            batch_register_tokens = self.register_tokens.expand(n, -1, -1)
            x = torch.cat([batch_register_tokens, x], dim=1)
        
        # Expand the class token to the full batch
        batch_class_tokens = self.class_tokens.expand(n, -1, -1)
        x = torch.cat([batch_class_tokens, x], dim=1)

        # Pass through the encoder
        x = self.encoder(x)

        # Get all class tokens and average them
        x = x[:, 0:self.num_class_tokens]
        x = torch.sum(x, dim=1)

        # Classification head
        x = self.head(x)

        return x


    def load_weights(
        self, 
        torch_pretrained_weights: Optional[str] = None, 
        timm_pretrained_weights: Optional[List] = None):
        """
        Loads pretrained weights into the model.

        Args:
            torch_pretrained_weights (str, optional): Path to the torch pretrained weights file or a URL to download from. Defaults to None.
            timm_pretrained_weights (List, optional): List containing the name of the timm model and the variant to load pretrained weights from. Defaults to None.
        
        Example:
            torch_pretrained_weights = 'ViT_B_16_Weights[IMAGENET1K_V1]'
            timm_pretrained_weights = ['facebookresearch/deit_base_patch16_224', 'deit_base_patch16_224']
            torch_pretrained_weights = 'path/to/torchweights.pth'
            timm_pretrained_weights = 'path/to/timmweights.pth'
        """
        
        # they cannot be both not None
        assert not (torch_pretrained_weights and timm_pretrained_weights), "You cannot load weights from both torch and timm at the same time."
        
        
        if torch_pretrained_weights is not None:
            print('Loading torch pretrained weights: ', torch_pretrained_weights)
            from .adapters import adapt_torch_state_dict
            if not os.path.exists(str(torch_pretrained_weights)):
                print('Downloading torch pretrained weights: ', torch_pretrained_weights)
                torch_pretrained_weights = eval(torch_pretrained_weights).get_state_dict(progress=False)
                adapted_state_dict = adapt_torch_state_dict(torch_pretrained_weights, num_classes=self.num_classes)
            else:
                torch_pretrained_weights = torch.load(torch_pretrained_weights)
                print(f'Loaded torch pretrained weights with these keys {list(torch_pretrained_weights.keys())}. I assume the model weights are in the the "model" key.')
                torch_pretrained_weights = torch_pretrained_weights['model']
                adapted_state_dict = adapt_torch_state_dict(torch_pretrained_weights, num_classes=self.num_classes)
            self.load_state_dict(adapted_state_dict, strict=False)
        elif timm_pretrained_weights is not None:
            print('Loading timm pretrained weights: ', timm_pretrained_weights)
            from .adapters import adapt_timm_state_dict
            if not os.path.exists(str(timm_pretrained_weights)):
                print('Downloading timm pretrained weights: ', timm_pretrained_weights)
                model = torch.hub.load(timm_pretrained_weights[0], timm_pretrained_weights[1], pretrained=True)
                timm_pretrained_weights = model.state_dict()
                del model
            else:
                timm_pretrained_weights = torch.load(timm_pretrained_weights)
                print(f'Loaded timm pretrained weights with these keys {list(timm_pretrained_weights.keys())}. I assume the model weights are in the the "model" key.')
                timm_pretrained_weights = timm_pretrained_weights['model']
            adapted_state_dict = adapt_timm_state_dict(timm_pretrained_weights, num_classes=self.num_classes)
            self.load_state_dict(adapted_state_dict, strict=False)
    
    