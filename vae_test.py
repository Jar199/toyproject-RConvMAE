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
from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
from pytorch_lightning.loggers import TensorBoardLogger
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader, Dataset
from dataset import VAEDataset
from experiment import VAEXperiment



parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)
        
tb_logger =  TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                               name=config['model_params']['name'],)

#model = VAEXperiment.load_from_checkpoint("logs/VanillaVAE/version_2/checkpoints/last.ckpt")(vae_models[config['model_params']['name']], config['exp_params'])
model = vae_models[config['model_params']['name']](**config['model_params'])
experiment = VAEXperiment(model,
                          config['exp_params'])
checkpoint = torch.load("logs/VanillaVAE/version_2/checkpoints/last.ckpt")
experiment.load_state_dict(checkpoint["state_dict"])
epochs = 3
data = VAEDataset(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)
data.setup()
for epoch in range(epochs):
    # x is data for 1 epoch
    # model
    test_input, test_label = next(iter(data.test_dataloader()))
    test_input = test_input.to(experiment.curr_device)
    test_label = test_label.to(experiment.curr_device)
    #recons = model.generate(x, labels = y) 
    recons = experiment.model.generate(test_input, labels = test_label)
    Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)

    vutils.save_image(recons.data,
                        os.path.join(tb_logger.log_dir , 
                                    "Reconstructions", 
                                    f"recons_{tb_logger.name}_Epoch_{epoch}.png"),
                        normalize=True,
                        nrow=12)
