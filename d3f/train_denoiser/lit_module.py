import cv2
import numpy as np
import math
import argparse
import random
import pytorch_lightning as pl
import segmentation_models_pytorch
import torch
import torch.nn as nn
import torch.optim as optimizers
import torch.optim.lr_scheduler as schedulers
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torchvision
from torch.utils.data import DataLoader
from d3f.dataset.image_dataset import ImageDataset

from kornia import augmentation as K
from kornia.augmentation import AugmentationSequential

from d3f.loss_functions import MseStructuralSimilarityLoss
from d3f.helpers import LoggingScheduler, convert_pyplot_figure_to_image_tensor

import matplotlib.pyplot as plt



class LitModule(pl.LightningModule):
    def __init__(self,**kwargs):
        super().__init__()
        
        self.save_hyperparameters()
            
        self.model = self.create_model_instance()
        self.training_criterion = MseStructuralSimilarityLoss(-1.0,1.0)

        self.shared_augmentation_sequence = self.create_shared_augmentation_sequence()

        self.image_logging_scheduler = LoggingScheduler()

    def create_model_instance(self):
        p = self.hparams

        encoder_name = p["encoder_name"]

        model = segmentation_models_pytorch.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=3,
            classes=3,
            activation=None,
        )
        return model

    def create_shared_augmentation_sequence(self):
        augmentation_sequence = AugmentationSequential(
            K.RandomAffine(
                degrees=10, 
                translate=[0.1, 0.1], 
                scale=[0.75, 1.25], 
                shear=10, 
                p=1.0,
            ),
        )
        return augmentation_sequence

    def train_dataloader(self):
        p = self.hparams
        return self.create_dataloader(p.input_image_list_path, p.mean, p.std)
    
    def create_dataloader(self, path, mean, std):
        p = self.hparams

        transform = nn.Sequential(
            T.Normalize(mean,std),
        )

        dataset = ImageDataset(
            path,
            transform=transform,
            )

        dataloader = DataLoader(
            dataset=dataset, 
            batch_size=p.batch_size,
            num_workers=p.num_workers,
            shuffle=True,
            )
        
        return dataloader

    def configure_optimizers(self):
        p = self.hparams

        optimizer = optimizers.Adam(self.model.parameters(), lr=p.learning_rate)

        scheduler = schedulers.CosineAnnealingLR(optimizer, T_max=p.cosine_scheduler_max_epoch)
 

        return [optimizer], [scheduler]

        return optimizer

    def forward(self,image):
        return self.model(image)

    def training_step(self, batch, batch_idx):

        self.image_logging_scheduler.update_with_step_number(self.global_step)

        image = batch["image"]

        image = self.shared_augmentation_sequence(image)

        image_noisy = self.blend_random_amount_of_noise_with_each_sample(image)

        image_prediction = self.model(image_noisy)
        
        loss = self.training_criterion(image_prediction, image)

        self.log_batch_as_image_grid(f"image", image)
        self.log_batch_as_image_grid(f"image_noisy", image_noisy)
        self.log_batch_as_image_grid(f"image_prediction", image_prediction)
        self.log(f"loss",loss)

        return loss

    def blend_random_amount_of_noise_with_each_sample(self,batch):
        p = self.hparams

        noise = torch.randn_like(batch)

        b,c,h,w = batch.shape

        r = self.sample_random_number_from_exponential_distribution(b,p.noise_exponential_sampling_lambda)

        noisy_batch = torch.sqrt(1-r) * batch + torch.sqrt(r)*noise

        return noisy_batch

    def sample_random_number_from_exponential_distribution(self,batch_size,lam):

        y = torch.rand(
            size=(batch_size,1,1,1),
            device=self.device,
        )

        c = 1/math.exp(lam)

        #use inverse sampling method
        x = 1/lam * torch.log( 1 / (y*(1-c) + c) )

        return x

    

    def log_batch_as_image_grid(self,tag, batch, first_batch_only=False):

        if self.image_logging_scheduler.should_we_log_this_step():

            nrows = 3
            ncols = 3
            n = nrows*ncols

            image = torchvision.utils.make_grid(batch[:n], nrows)

            image *= 0.5
            image += 0.5
            image = image.clamp(0,1)

            self.logger.experiment.add_image( tag, image, self.global_step)


    