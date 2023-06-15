import os
import yaml
import argparse
import math
from pathlib import Path
import torch
from torch import optim
from models import BaseVAE
from models.types_ import *
from models import *
#from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
from pytorch_lightning.loggers import TensorBoardLogger
#import torchvision.utils as vutils
#from torchvision.datasets import CelebA
#from torch.utils.data import DataLoader, Dataset
#from dataset import VAEDataset
from experiment import VAEXperiment
import torchvision


class VAELoader():
    def __init__(self,):
        filename = 'configs/vae.yaml'
        with open(filename, 'r') as file:
            try:
                config = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)
                
        self.tb_logger =  TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                                    name=config['model_params']['name'],)

        #model = VAEXperiment.load_from_checkpoint("logs/VanillaVAE/version_2/checkpoints/last.ckpt")(vae_models[config['model_params']['name']], config['exp_params'])
        self.model = vae_models[config['model_params']['name']](**config['model_params'])
        self.experiment = VAEXperiment(self.model,
                                config['exp_params'])
        checkpoint = torch.load("logs/VanillaVAE/version_1/checkpoints/last.ckpt")
        self.experiment.load_state_dict(checkpoint["state_dict"])
        
    def test(self, test_input):#, test_label): # testing VAE for 1 epoch
        #test_input, test_label = next(iter(data.test_dataloader()))
        #test_input = test_input.to(self.experiment.curr_device)
        #test_label = test_label.to(self.experiment.curr_device)
        
        #recons = self.experiment.model.generate(test_input, labels = test_label)
        test_input = torchvision.transforms.Resize(64)
        recons = self.experiment.model.generate(test_input)
        
        return recons.data