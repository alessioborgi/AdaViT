import os
import wandb
from pprint import pprint
import time
import yaml

last_print_time = 0


class SimpleLogger:
    """
    Simple logger for logging to stdout and to a file.
    """
    def __init__(self, settings, dir, log_every:int =60):
        self.log_every = log_every
        self.log_file_path = os.path.join(dir, 'log.txt')
        os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
        self.log_file = open(self.log_file_path, 'a+')
        self.log(settings[0])

        print('Logging to', self.log_file_path)
        print('This local logger is not recommended for large scale experiments. Use wandb instead.')
        
    
    def log(self, args):
        global last_print_time
        current_time = time.time()
    
        # Check if it's been at least a minute since the last print
        if current_time - last_print_time >= self.log_every:
            last_print_time = current_time 
            pprint(args)
        print(args, file=self.log_file)
        self.log_file.flush()

    
    def close(self):
        self.log_file.close()


class WandbLogger:
    """
    Logger for logging to wandb.
    """
    def __init__(self, wandb_entity, wandb_project, settings, train_config_path, dir=None, wandb_run=None):
        self.entity = wandb_entity
        self.project = wandb_project
        self.wandb_run = wandb_run
        self.config = settings if isinstance(settings, dict) else eval(settings)
        
                
        wandb.init(
            entity=wandb_entity,
            project=wandb_project,
            config=self.config,
            name=wandb_run,
            dir=dir,
        )

    settings = gather_settings_train(self.train_config_path)
    print("The settings is: ", settings)
    
    wandb.init(
    # set the wandb project where this run will be logged
    project="Laplacian_AdaViT",

    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.02,
    "architecture": "CNN",
    "dataset": "CIFAR-100",
    "epochs": 10,
    }
)
    
    def log(self, dict_to_log):
        wandb.log(dict_to_log)
    
    def close(self):
        wandb.finish()
    
    def gather_settings_train(train_config_path):
        settings = {}

        # Load contents of train_config.yaml into settings
        with open(train_config_path, "r") as file:
            train_config_dict = yaml.safe_load(file)
            settings.update(train_config_dict)

        # Helper function to recursively gather settings from defaults
        def gather_defaults(defaults_list):
            for default in defaults_list:
                if isinstance(default, str):
                    # If default is a string, assume it's a path to a YAML file
                    default_path = os.path.join(os.path.dirname(train_config_path), default)
                    with open(default_path, "r") as file:
                        default_dict = yaml.safe_load(file)
                        settings.update(default_dict)
                elif isinstance(default, dict):
                    # If default is a dictionary, recursively gather its settings
                    gather_defaults(default.values())

        gather_defaults(settings.get("defaults", []))

        return settings
    
