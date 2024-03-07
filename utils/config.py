import os
import torch
import torch.nn as nn
import yaml

class CFG:
    def __init__(self, config_file='./configs/config.yaml'):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)

        self.classes = config['classes']
        self.path = config['path']
        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']
        self.split = config['split']
        self.input_size = config['input_size']
        self.num_epochs = config['num_epochs']
        self.device = config['device']
        self.seed = config["seed"]
        self.debug = config["debug"]
        
        if self.debug:
            self.num_epochs = 5
        
        if not torch.cuda.is_available():
            self.device = "cpu"

CFG = CFG()

