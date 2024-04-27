from collections import defaultdict
from omegaconf import OmegaConf,DictConfig
import torch
from .utils import get_model_device, get_forward_masks
from einops import reduce
from torch.nn.functional import cross_entropy, relu
from torch.special import entr
from abc import ABC, abstractmethod
from typing import List, Literal, Optional
from hydra.utils import instantiate
import numpy as np


"""
A set of functions and classes for computing different types of losses.
Class implementations subclass the abstract class ModelLoss, which requires the implementation of a forward method.
ModelLoss subclasses are used to compute model specific regularizations, so they receive the model as input and return a loss value.

Multiple losses can be composed together using the LossCompose class, which takes a dictionary of loss functions and their arguments as input.
See the config/loss files to see how to use the LossCompose class.

"""


class ModelLoss(torch.nn.Module, ABC):
    
    @abstractmethod
    def forward(self, model, **kwargs):
        pass


####################################################### functional implementations ##################################################################


def sparsity_loss_per_block(model, budget: float = 0.65, sparsity_type : Literal['l1', 'mse', 'cross_entropy' ] = 'l1', **kwargs):
    """
    Computes the sparsity loss per block.

    Args:
        model: The model for which to compute the sparsity loss.
        budget (float): The desired sparsity level.
        sparsity_type (str): The type of sparsity loss to compute, 
        **kwargs: Additional keyword arguments.

    Returns:
        Tuple: A tuple containing the mean sparsity loss and the mean intra-entropy.
    """
    
    masks = get_forward_masks(model)

    # compute the sparsity loss
    sparsity_loss = []
    intra_entropy = []
    for _, mask in masks.items():
        # compute the sparsity loss for each sequence in the batch 
        # mask only contains 1s and 0s, so the mean is the sparsity
        sparsity = reduce(mask, 'b s 1 -> b', 'mean') # this is basically the percentage of 1s in the mask
        
        if sparsity_type == 'mse':
            # force sparsity to be close to budget with mse
            sparsity_loss.append(torch.mean((sparsity - budget) ** 2))
        if sparsity_type == 'cross_entropy':
            # force sparsity to be close to budget with cross entropy
            sparsity_loss.append(cross_entropy(sparsity, torch.tensor([budget] * sparsity.shape[0]).to(get_model_device(model))))
        elif sparsity_type == 'l1':
            # force sparsity to be close to budget with l1
            sparsity_loss.append(torch.mean(torch.abs(sparsity - budget)))

        # force sparsity inside image by maximizing entropy
        # print(sparsity)
        intra_entropy.append(entr(sparsity))
        
    sparsity_loss = torch.stack(sparsity_loss)


    return torch.mean(sparsity_loss)


def entropy_per_blocks(model, **kwargs):

    # get all masks from the model, each mask is a tensor of shape (batch_size, sequence_len, 1)
    masks = get_forward_masks(model)

    # compute the sparsity loss
    intra_entopy = []
    for _, mask in masks.items():
        # compute the sparsity loss for each mask
        sparsity = reduce(mask, 'b s 1 -> b', 'mean')
        intra_entopy.append(entr(sparsity))
    
    intra_entopy = torch.stack(intra_entopy)

    return torch.mean(intra_entopy), 


def solo_l1(model, budget: float = 0.25, strict:bool = False, **kwargs):

    # get all masks from the model, each mask is a tensor of shape (batch_size, sequence_len, 1)
    masks = get_forward_masks(model)

    # iterate over masks
    sparsity_loss = []
    for _, mask in masks.items():
        sparsity = reduce(mask, 'b s 1 -> b', 'mean') # this is basically the percentage of 1s in the mask
        sparsity_loss.append(torch.sum(torch.abs(sparsity - budget)))
    
    sparsity_loss = torch.stack(sparsity_loss)

    return torch.mean(sparsity_loss)


def solo_mse(model, 
             budget: float = 0.65, 
             strict: bool = False, 
             skip_layers: List = [],  
             per_layer: bool = True,
             **kwargs):
    
    # get all masks from the model, each mask is a tensor of shape (batch_size, sequence_len, 1)
    masks = get_forward_masks(model)

    # iterate over masks
    sparsity_loss = []
    for layer, (_, mask) in enumerate(masks.items()):
        if layer not in skip_layers: 
            sparsity = reduce(mask, 'b s 1 -> b', 'mean') # this is basically the percentage of 1s in the mask, for each image in the batch
            
            # if per layer, we compute MSE for each layer 
            if per_layer:
                sparsity = torch.sum((sparsity - budget) ** 2 if strict else (relu(sparsity - budget))**2)
            sparsity_loss.append(sparsity)
             
    
    sparsity_loss = torch.stack(sparsity_loss)


    if not per_layer:
        # if not per layer, we average the sparsity across all layers and then compute the MSE 
        sparsity_loss = torch.mean(sparsity_loss)
        sparsity_loss = torch.sum((sparsity_loss - budget) ** 2 if strict else (relu(sparsity_loss - budget))**2)

    return torch.mean(sparsity_loss * (2-budget))


def l1_and_intraentropy(model, budget: float = 0.65,  **kwargs):

    # get all masks from the model, each mask is a tensor of shape (batch_size, sequence_len, 1)
    masks = get_forward_masks(model)

    # iterate over masks
    sparsity_loss = []
    intra_entropy = []
    for _, mask in masks.items():
        sparsity = reduce(mask, 'b s 1 -> b', 'mean')
        sparsity_loss.append(torch.sum(torch.abs(relu(sparsity - budget))))
    
        intra_entropy.append(entr(sparsity))
    
    sparsity_loss = torch.stack(sparsity_loss)

    return torch.mean(sparsity_loss)


def avit_ponder_loss(model, **kwargs):
    """
    Computes the ponder loss of the model.

    Args:
        model: The model for which to compute the ponder loss.
        **kwargs: Additional keyword arguments.

    Returns:
        torch.Tensor: The ponder loss.
    """
    ponder_loss = torch.mean(model.encoder.rho_token)

    return ponder_loss


def avit_distr_prior_loss_univariate(model, target_depth, scaling, **kwargs):
    """
    Computes the distribution prior loss of the model.

    Args:
        model: The model for which to compute the distribution prior loss.
        log_distr_target: The target distribution to compare the model's distribution to.
        **kwargs: Additional keyword arguments.

    Returns:
        torch.Tensor: The distribution prior loss.
    """
    
    # Gaussian_Distribution
    target_dist = torch.distributions.Normal(loc=target_depth, scale=scaling)
    
    # Laplace_Distribution
    #target_dist = torch.distributions.Laplace(loc=target_depth, scale=scaling/ (2 ** 0.5))
    
    # Create a Student's t-distribution with specified degrees of freedom
    #degrees_of_freedom = 30  # Adjust this parameter to control tail thickness
    #target_dist = torch.distributions.StudentT(loc=target_depth, scale=scaling/ (2 ** 0.5), df=degrees_of_freedom)
    
    # Cauchy Distribution
    #target_dist = torch.distributions.Cauchy(loc=target_depth, scale=scaling/ (2 ** 0.5))

    target_dist = target_dist.log_prob(torch.arange(model.num_layers) + 1)
    halting_score_distr = torch.stack(model.encoder.halting_score_layer)
    halting_score_distr = halting_score_distr / torch.sum(halting_score_distr)
    halting_score_distr = torch.clamp(halting_score_distr, 0.001, 0.999)
    
    distr_prior_loss = torch.nn.functional.kl_div(halting_score_distr.log(),
                                                    target_dist.to(halting_score_distr.device).detach(),
                                                    reduction='batchmean',
                                                    log_target=True)

    return  distr_prior_loss



    
# Multivariate Distribution Prior Loss implementation.    
def avit_distr_prior_loss(model, target_depth=[5, 10], scaling=[None, None], covariance_matrices=None, **kwargs):
    """
    Computes the multivariate distribution prior loss of the model.

    Args:
        model: The model for which to compute the distribution prior loss.
        target_depth: The target depths for each layer in the model.
        scaling: The scaling factors for each layer in the model.
        covariance_matrices: The covariance matrices for the multivariate distribution.
        **kwargs: Additional keyword arguments.

    Returns:
        torch.Tensor: The multivariate distribution prior loss.
    """
    
    # Convert target_depth to tensor
    target_depth = torch.tensor(target_depth, dtype=torch.float32)
    
    
    # Gaussian Multivariate Distribution
    
    # Identity Covariance Matrix.
    if (covariance_matrices is None) or (covariance_matrices.lower() in {"identity", "id", "i"}):
        covariance_matrices = torch.eye(len(target_depth), dtype=torch.float32)  
        
    # Strong Covariance Matrix.
    # Positive Correlation (Earlier Halting in Deeper Layers)
    elif covariance_matrices.lower() in {"strong", "s", "positive", "poscorr", "hard", "early", "earlier"}:
        covariance_matrices = torch.eye(len(target_depth), dtype=torch.float32)  
        for i in range(len(target_depth) - 1):
            for j in range(i + 1, len(target_depth)):
                covariance_matrices[i, j] = 0.5  # Adjust value (0 to 1) based on desired correlation 
                covariance_matrices[j, i] = 0.5  # Adjust value (0 to 1) based on desired correlation strength
    
    # Soft Covariance Matrix.
    # Negative Correlation (Later Halting in Deeper Layers)
    elif covariance_matrices.lower() in {"soft", "sf", "negative", "negcorr", "late", "later"}:
        covariance_matrices = torch.eye(len(target_depth), dtype=torch.float32)  
        for i in range(len(target_depth) - 1):
            for j in range(i + 1, len(target_depth)):
                covariance_matrices[i, j] = -0.5  # Adjust value (0 to 1) based on desired correlation 
                covariance_matrices[j, i] = -0.5  # Adjust value (0 to 1) based on desired correlation strength
    
    
    # Convert scaling to tensor if it's not None and instantiate multivariate distribution.
    if scaling is not None:
        scaling = torch.tensor(scaling, dtype=torch.float32)
        scaling_diagonal = torch.diag_embed(scaling)
        target_dist = torch.distributions.MultivariateNormal(target_depth, scaling_diagonal @ covariance_matrices)
    else:
        target_dist = torch.distributions.MultivariateNormal(target_depth, covariance_matrices)
        

    # Compute the log probabilities of the target depths
    target_dist_log_prob = target_dist.log_prob(torch.arange(model.num_layers, dtype=torch.float32).repeat(len(target_depth), 1).t() + 1)

    # Get halting scores distribution from the model
    halting_score_distr = torch.stack(model.encoder.halting_score_layer)
    halting_score_distr = halting_score_distr / torch.sum(halting_score_distr)
    halting_score_distr = torch.clamp(halting_score_distr, 0.001, 0.999)
    
    # Compute the Kullback-Leibler divergence between the halting scores distribution and the target distribution
    distr_prior_loss = torch.nn.functional.kl_div(halting_score_distr.log(),
                                                   target_dist_log_prob.to(halting_score_distr.device).detach(),
                                                   reduction='batchmean',
                                                   log_target=True)

    return distr_prior_loss








####################################################### class implementations ##################################################################


class AViTPonderLoss(ModelLoss):
    """
    Computes the ponder loss of the model.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, model, **kwargs):
        """
        Computes the ponder loss of the model.

        Args:
            model (nn.Module): The model to compute the ponder loss for.
            **kwargs: Additional arguments.

        Returns:
        torch.Tensor: The ponder loss.
        """
        return avit_ponder_loss(model)


class AViTDPriorLoss(ModelLoss):
    """
    Computes the distribution prior loss of the model.
    """

    def __init__(self, target_depth: int, scaling: Optional[float] = None, covariance_matrices: Optional[float] = None,) -> None:
        super().__init__()
        self.target_depth = target_depth
        self.scaling = scaling
        self.covariance_matrices = covariance_matrices

    def forward(self, model, **kwargs):
        """
        Computes the distribution prior loss of the model.

        Args:
            model (nn.Module): The model to compute the distribution prior loss for.
            **kwargs: Additional arguments.

        Returns:
            torch.Tensor: The distribution prior loss.
        """
        return avit_distr_prior_loss(model, target_depth=self.target_depth, scaling=self.scaling, covariance_matrices=self.covariance_matrices)


class AViTDPriorLossMultivariate(ModelLoss):
    """
    Computes the distribution prior loss of the model.
    """

    def __init__(self, target_depth: List[int], scaling: Optional[List[int]] = None) -> None:
        super().__init__()
        self.target_depth = target_depth
        self.scaling = scaling

    def forward(self, model, **kwargs):
        """
        Computes the distribution prior loss of the model.

        Args:
            model (nn.Module): The model to compute the distribution prior loss for.
            **kwargs: Additional arguments.

        Returns:
            torch.Tensor: The distribution prior loss.
        """
        return avit_distr_prior_loss(model, target_depth=self.target_depth, scaling=self.scaling)

    
    
class SparsityLoss(ModelLoss):
    """
    Computes the sparsity loss of the model.
    """

    def __init__(self, budget: float) -> None:
        super().__init__()
        self.budget = budget

    def forward(self, model, budget=None, **kwargs):
        """
        Computes the sparsity loss of the model.

        Args:
            model (nn.Module): The model to compute the sparsity loss for.
            **kwargs: Additional arguments.

        Returns:
            torch.Tensor: The sparsity loss.
        """
        return sparsity_loss_per_block(model, budget= budget or self.budget, **kwargs)


class EntropyLoss(ModelLoss):
    """
    Computes the entropy loss of the model.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, model, **kwargs):
        """
        Computes the entropy loss of the model.

        Args:
            model (nn.Module): The model to compute the entropy loss for.
            **kwargs: Additional arguments.

        Returns:
            torch.Tensor: The entropy loss.
        """
        return entropy_per_blocks(model)


class L1Loss(ModelLoss):
    """
    Computes the L1 loss of the model.
    """

    def __init__(self, budget: float) -> None:
        super().__init__()
        self.budget = budget

    def forward(self, model, budget = None, **kwargs):
        """
        Computes the L1 loss of the model.

        Args:
            model (nn.Module): The model to compute the L1 loss for.
            **kwargs: Additional arguments.

        Returns:
            torch.Tensor: The L1 loss.
        """
        # if batch is not None and a budget was not provided, compute the budget from the batch
        return solo_l1(model, budget or self.budget)


class MSELoss(ModelLoss):
    """
    Computes the MSE loss of the model.
    """

    def __init__(self, budget: float = None, strict: bool = False, skip_layers : List = [], per_layer:bool = True, **kwargs) -> None:
        super().__init__()
        self.budget = budget
        self.strict = strict
        self.skip_layers = skip_layers
        self.per_layer = per_layer

    def forward(self, model, budget = None, per_layer: bool = None, **kwargs):
        """
        Computes the MSE loss of the model.

        Args:
            model (nn.Module): The model to compute the MSE loss for.
            **kwargs: Additional arguments.

        Returns:
            torch.Tensor: The MSE loss.
        """
        assert budget is not None or self.budget is not None, 'budget must be provided either as argument or as class attribute'
        per_layer = per_layer if per_layer is not None else self.per_layer
        return solo_mse(model, budget if budget is not None else self.budget, self.strict, skip_layers=self.skip_layers, per_layer=per_layer)


class ChannelMSELoss(ModelLoss):
    """
    Computes the MSE loss of the model. It is a copy of MSELoss with a different name. 
    The reason for the different name is that this loss is supposed to be used for channel bandwith and not for general model budget.
    """

    def __init__(self, budget: float = None, strict: bool = False, skip_layers : List = [], **kwargs) -> None:
        super().__init__()
        self.budget = budget
        self.strict = strict
        self.skip_layers = skip_layers

    def forward(self, model, channel_budget = None, **kwargs):
        """
        Computes the MSE loss of the model.

        Args:
            model (nn.Module): The model to compute the MSE loss for.
            **kwargs: Additional arguments.

        Returns:
            torch.Tensor: The MSE loss.
        """
        assert channel_budget or self.budget, 'budget must be provided either as argument or as class attribute'
        
        return solo_mse(model, channel_budget if channel_budget is not None else self.budget, self.strict, skip_layers=self.skip_layers)


class L1AndIntraEntropyLoss(ModelLoss):
    """
    Computes the L1 loss and the intra-entropy of the model.
    """

    def __init__(self, budget: float) -> None:
        super().__init__()
        self.budget = budget

    def forward(self, model, budget = None, **kwargs):
        """
        Computes the L1 loss and the intra-entropy of the model.

        Args:
            model (nn.Module): The model to compute the L1 loss and the intra-entropy for.
            **kwargs: Additional arguments.

        Returns:
            Tuple: A tuple containing the L1 loss and the intra-entropy.
        """
        return l1_and_intraentropy(model, budget or self.budget)


class AlwaysZeroLoss(ModelLoss):
    """
    A loss function that always returns zero.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, model, **kwargs):
        return torch.tensor(0.0), torch.tensor(0.0)



#########################################################################################################################


class LossCompose:
    """
    A class that composes multiple loss functions together.

    Args:
        losses_dict (dict): A dictionary containing the loss functions and their arguments.
        Notice that each element in the dictionary must be a dictionary containing 
        at least the key _target_ that points to a class that can be instantiated by hydra. 

    Attributes:
        additional_losses (defaultdict): A dictionary that stores the additional losses with their weights and loss functions. See losses yaml file for example.

    Methods:
        compute: Computes the total loss by evaluating each additional loss function.

    """

    def __init__(self, losses_dict):
        """
        Initializes the LossCompose object.

        Args:
            losses_dict (dict): A dictionary containing the loss functions and their arguments.

        """
        if isinstance(losses_dict, DictConfig):
            losses_dict = OmegaConf.to_container(losses_dict, resolve=True)

        self.additional_losses = defaultdict(dict)  
        for loss, loss_args in losses_dict.items():
            self.additional_losses[loss]['weight'] = loss_args.pop('weight', 1.)
            self.additional_losses[loss]['loss_fn'] = instantiate(loss_args)
        

    def compute(self, model, dict_prefix='', return_dict=True, **kwargs):
        """
        Computes the total loss by evaluating each additional loss function.

        Args:
            model: The model used for computing the loss.
            dict_prefix (str): A prefix to be added to the loss names in the losses_dict.
            return_dict (bool): Whether to return the losses_dict along with the total loss.
            **kwargs: Additional keyword arguments to be passed to the loss functions.

        Returns:
            total_loss: The computed total loss.
            (optional) losses_dict: A dictionary containing the individual losses and their values. 

        """
        losses_dict = {}
        total_loss = []
        for loss, loss_args in self.additional_losses.items():
            l = loss_args['loss_fn'](model, **kwargs) * loss_args['weight']
            losses_dict[f'{dict_prefix}{loss}'] = l.detach().item()
            total_loss.append(l)
        total_loss = torch.stack(total_loss).sum()
        if return_dict:
            return losses_dict, total_loss
        else:
            return total_loss
