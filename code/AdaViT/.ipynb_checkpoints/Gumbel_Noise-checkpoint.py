import torch

def add_gumbel_noise(logits, eps=1e-20, device='cuda'):
    """
    Function to sample Gumbel noise and add it to the logits.

    Parameters:
    - logits (torch.Tensor): Input logits.
    - eps (float): Small value to prevent numerical instability.
    - device (str): Device to create the noise tensor.

    Returns:
    torch.Tensor: Logits with added Gumbel noise.
    """
    u = torch.empty(logits.size(), device=device).uniform_(0, 1)
    gumbel_noise = -torch.log(-torch.log(u + eps) + eps)
    return logits + gumbel_noise
