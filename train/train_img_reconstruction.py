AdaViT.import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from torchvision import transforms as T
from torch.utils.data import DataLoader
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import torchmetrics
import hydra
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate
from pprint import pprint
from torch.utils.data import Subset



from peekvit.utils.utils import get_checkpoint_path, save_state, load_state, make_experiment_directory
from peekvit.models.topology import reinit_class_tokens, train_only_these_params
from peekvit.utils.losses import LossCompose
from peekvit.utils.visualize import plot_masked_images





@hydra.main(version_base=None, config_path="../configs", config_name="train_config_personal")
def train(cfg: DictConfig):

    torch.manual_seed(cfg.seed)

    # experiment name and settings
    exp_name = cfg.experiment_name
    device = torch.device(cfg.device)
    experiment_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    experiment_dir, checkpoints_dir = make_experiment_directory(experiment_dir)
    
    
    # logger
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    pprint(config_dict)
    logger = instantiate(cfg.logger, settings=str(config_dict), dir=experiment_dir)
    

    # dataset and dataloader
    training_args = cfg.training
    dataset = instantiate(cfg.dataset)
    train_dataset, val_dataset = dataset.train_dataset, dataset.val_dataset
    train_loader = DataLoader(train_dataset, batch_size=training_args.train_batch_size, shuffle=True, num_workers=training_args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=training_args.eval_batch_size, shuffle=False, num_workers=training_args.num_workers, pin_memory=True)

    # model
    model = instantiate(cfg.model)
    model.to(device)

    # load from checkpoint if requested
    load_from = cfg.load_from
    if load_from is not None:
        # load from might be a path to a checkpoint or a path to an experiment directory, handle both cases
        load_from = load_from if load_from.endswith('.pth') else get_checkpoint_path(load_from)
        print('Loading model from checkpoint: ', load_from)
        model, _, _, _, _ = load_state(load_from, model=model)
    
    # edit model here if requested
    if training_args['reinit_class_tokens']:
        model = reinit_class_tokens(model)
    


    # main loss 
    main_criterion = instantiate(cfg.loss.classification_loss)
    
    # we might have N additional losses
    # so we store the in a dictionary
    additional_losses = None
    if cfg.loss.additional_losses is not None:
        additional_losses = LossCompose(cfg.loss.additional_losses)
        
    # metrics
    metric = torchmetrics.classification.Accuracy(task="multiclass", num_classes=cfg.model.num_classes).to(device)

    # optimizer and scheduler
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    scheduler = None
    if 'scheduler' in cfg:
        scheduler = instantiate(cfg.scheduler, optimizer=optimizer)

    # training loop
    def train_epoch(model, loader, optimizer, epoch):
        model.train()
        if not training_args['train_backbone']:
            model = train_only_these_params(model, ['gate', 'class', 'head', 'threshold', 'budget'], verbose=epoch==0)

        for batch, labels in tqdm(loader, desc=f'Training epoch {epoch}'):
            batch, labels = batch.to(device), labels.to(device)
            optimizer.zero_grad()
            out, reconstructed_input, mask = model(batch)
            
            main_loss = main_criterion(out, labels) 
            reconstruction_loss = torch.mean(((batch-reconstructed_input)**2)*mask) # TODO only compute loss on pixels that are masked
            add_loss_dict, add_loss_val = {}, 0.0
            if additional_losses is not None:
                add_loss_dict, add_loss_val = additional_losses.compute(
                    model, 
                    budget=model.current_budget,
                    channel_budget=getattr(model, 'current_channel_budget', None), 
                    dict_prefix='train/')
            loss = main_loss + add_loss_val + reconstruction_loss
            loss.backward()
            # Apply gradient clipping
            if training_args['clip_grad_norm'] is not None:
                clip_grad_norm_(model.parameters(), max_norm=training_args['clip_grad_norm'])
            optimizer.step()
            logger.log({'train/total_loss': loss.detach().item(), 'train/classification_loss': main_loss.detach().item(), 'train/reconstruction_loss':reconstruction_loss.detach().item()} | add_loss_dict)
        if scheduler:
            logger.log({'train/lr': scheduler.get_last_lr()[0]})
            scheduler.step()

    @torch.no_grad()
    def validate_epoch(model, loader, epoch, budget=''):
        model.eval()
        batches_loss = 0
        for batch, labels in tqdm(loader, desc=f'Validation epoch {epoch} {budget}'):
            batch, labels = batch.to(device), labels.to(device)
            out, _, _ = model(batch)
            val_loss = main_criterion(out, labels) 
            predicted = torch.argmax(out, 1)
            metric(predicted, labels)
            batches_loss += val_loss.detach().item()
        
        val_loss = batches_loss / len(loader)
        acc = metric.compute()
        metric.reset()

        return acc, val_loss

    # validation loop
    @torch.no_grad()
    def validate(model, loader, epoch):
        val_budgets = cfg.training.val_budgets or [1.]
        model.eval()
        if hasattr(model, 'set_budget'):
            for budget in val_budgets:
                model.set_budget(budget)
                acc, val_loss = validate_epoch(model, loader, epoch, budget=f'budget_{budget}')
                logger.log({f'budget_{budget}/val/accuracy': acc, f'budget_{budget}/val/loss': val_loss})
        else:
            acc, val_loss = validate_epoch(model, loader, epoch)
            logger.log({'val/accuracy': acc, 'val/loss': val_loss})
        
        return acc, val_loss


    # Aux function to plot masks during training   
    # Assumes model has budget 
    def plot_masks_in_training(model, budgets):
        
        subset_idcs = torch.arange(0, len(val_dataset), len(val_dataset)//training_args['num_images_to_plot'])
        images_to_plot = Subset(val_dataset, subset_idcs)
        hard_prefix = 'hard_'

        for budget in budgets:

            model.set_budget(budget)
            
            images = plot_masked_images(
                            model,
                            images_to_plot,
                            model_transform=None,
                            visualization_transform=dataset.denormalize_transform,
                            hard=True,
                        )
            
            os.makedirs(f'{experiment_dir}/images/epoch_{epoch}', exist_ok=True)
            os.makedirs(f'{experiment_dir}/images/epoch_{epoch}/budget_{budget}', exist_ok=True)
            for i, (_, img) in enumerate(images.items()):
                img.savefig(f'{experiment_dir}/images/epoch_{epoch}/budget_{budget}/{hard_prefix}{subset_idcs[i]}.png')
    

    def plot_reconstructed_images_in_training(model, budgets):
        
        subset_idcs = torch.arange(0, len(val_dataset), len(val_dataset)//training_args['num_images_to_plot'])
        images_to_plot = Subset(val_dataset, subset_idcs)
        from peekvit.utils.visualize import plot_reconstructed_images
        for budget in budgets:

            model.set_budget(budget)

            images = plot_reconstructed_images(
                            model,
                            images_to_plot,
                            model_transform=None,
                            visualization_transform=dataset.denormalize_transform,
                        )
            
            os.makedirs(f'{experiment_dir}/images/epoch_{epoch}', exist_ok=True)
            os.makedirs(f'{experiment_dir}/images/epoch_{epoch}/reconstructed_budget_{budget}', exist_ok=True)
            for i, (_, img) in enumerate(images.items()):
                img.savefig(f'{experiment_dir}/images/epoch_{epoch}/reconstructed_budget_{budget}/reconstructed_img_{subset_idcs[i]}.png')
        
    
    # Training
    for epoch in range(training_args['num_epochs']+1):

        train_epoch(model, train_loader, optimizer, epoch)
        
        if training_args['eval_every'] != -1 and epoch % training_args['eval_every'] == 0:
            validate(model, val_loader, epoch)
            
        if training_args['checkpoint_every'] != -1 and epoch % training_args['checkpoint_every'] == 0:
            save_state(checkpoints_dir, model, cfg.model, cfg.noise, optimizer, epoch)
        
        if training_args['plot_masks_every'] != -1 and epoch % training_args['plot_masks_every'] == 0:
            if hasattr(model, 'set_budget'):
                plot_masks_in_training(model, cfg.training.val_budgets)
            else:
                print('[WARNING] Plotting masks is only supported for models with a budget. Skipping...')
        
        if training_args['plot_reconstructed_images_every'] != -1 and epoch % training_args['plot_reconstructed_images_every'] == 0:
            plot_reconstructed_images_in_training(model, cfg.training.val_budgets)
            
            



    
if __name__ == '__main__':
    train()