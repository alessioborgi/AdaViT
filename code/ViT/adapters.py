import torch
from torch import nn
import re


def adapt_torch_state_dict(torch_state_dict, num_classes:int):
    """
    Adapt the weights of a Pytorch Vision Transformer state dictionary to a VisionTransformer as defined in this repository. 
    Possibly edit the head to match the number of classes.

    Possible state dicts to be passes are found at https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py
    
    Args:
        state_dict (nn.Module): The state dictionary containing the weights of a Pytorch Vision Transformer.
        num_classes (int): The number of classes for which to adapt the weights.
        
    Returns:
        dict: The adapted state dictionary with updated keys.
    """
    
    new_state_dict = {}
    def adapt_param_name(param):
        p = param.replace('mlp.0', 'mlp.fc1').replace('mlp.3', 'mlp.fc2').replace('heads.head', 'head')
        p = p.replace('mlp.linear_1', 'mlp.fc1').replace('mlp.linear_2', 'mlp.fc2')
        if p.count('self_attention') == 1:
            p = p.replace('self_attention', 'self_attention.self_attention')
        
        if p == 'class_token':
            return 'class_tokens'
        
        p = re.sub(r'encoder_layer_(\d)', r'\1', p)
        return p

    for param_name, param in torch_state_dict.items():
        new_param_name = adapt_param_name(param_name)
        new_state_dict[new_param_name] = param
    
    # if num classes is different from the original, replace the head with a randomly initialized one
    old_head_shape = new_state_dict['head.weight'].shape
    if old_head_shape[0] != num_classes:
        print('Loading weights for a different number of classes. Replacing head with random weights. You should fine-tune the model.')
        new_head_shape = (num_classes, old_head_shape[1])
        new_state_dict['head.weight'] = torch.zeros(new_head_shape)
        new_state_dict['head.bias'] = torch.zeros(num_classes)
    
    return new_state_dict


def adapt_timm_state_dict(timm_state_dict, num_classes:int):
    """
    Adapt the weights of a Timm Vision Transformer state dictionary to a VisionTransformer as defined in this repository.
    Possibly edit the head to match the number of classes.

    Args:
        state_dict (nn.Module): The state dictionary containing the weights of a Pytorch Vision Transformer.
        num_classes (int): The number of classes for which to adapt the weights.

    Returns:
        dict: The adapted state dictionary with updated keys.
    """

    new_state_dict = {}
    def adapt_param_name(p):

        p = p.replace('norm1', 'ln_1').replace('norm2', 'ln_2')
        p = p.replace('attn.qkv.bias', 'self_attention.self_attention.in_proj_bias')
        p = p.replace('attn.qkv.weight', 'self_attention.self_attention.in_proj_weight')
        p = p.replace('attn.proj.bias', 'self_attention.self_attention.out_proj.bias')
        p = p.replace('attn.proj.weight', 'self_attention.self_attention.out_proj.weight')

        p = p.replace('patch_embed.proj.bias', 'conv_proj.bias')
        p = p.replace('patch_embed.proj.weight', 'conv_proj.weight')

        p = p.replace('cls_token', 'class_tokens')

        p = p.replace('pos_embed', 'encoder.pos_embedding' )

        p = p.replace('norm.weight', 'encoder.ln.weight')
        p = p.replace('norm.bias', 'encoder.ln.bias')


        p =  re.sub(r'blocks.(\d+)', r'encoder.layers.\1', p)
        return p

    for param_name, param in timm_state_dict.items():
        new_param_name = adapt_param_name(param_name)
        new_state_dict[new_param_name] = param

    # if num classes is different from the original, replace the head with a randomly initialized one
    old_head_shape = new_state_dict['head.weight'].shape
    if old_head_shape[0] != num_classes:
        print('Loading weights for a different number of classes. Replacing head with random weights. You should fine-tune the model.')
        new_head_shape = (num_classes, old_head_shape[1])
        new_state_dict['head.weight'] = torch.zeros(new_head_shape)
        new_state_dict['head.bias'] = torch.zeros(num_classes)

    return new_state_dict
