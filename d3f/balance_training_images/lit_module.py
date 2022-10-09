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
from d3f.helpers import LoggingScheduler



class LitModule(pl.LightningModule):
    def __init__(self,**kwargs):
        super().__init__()
        
        self.save_hyperparameters()
            
        self.model = self.create_model_instance()
        self.training_criterion = MseStructuralSimilarityLoss(-1.0,1.0)

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

    def train_dataloader(self):
        p = self.hparams
        return self.create_dataloader(p.data_path, p.mean, p.std)
    
    def val_dataloader(self):
        p = self.hparams
        return self.create_dataloader(p.data_path, p.mean, p.std)
   

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

        return optimizer

    def training_step(self, batch, batch_idx):
        
        self.image_logging_scheduler.update_with_step_number(self.global_step)

        image = batch["image"]

        image_noisy = self.blend_fixed_amount_of_noise_with_each_sample(image)

        image_prediction = self.model(image_noisy)
        
        loss = self.training_criterion(image_prediction, image)

        self.log_batch_as_image_grid(f"image", image)
        self.log_batch_as_image_grid(f"image_noisy", image_noisy)
        self.log_batch_as_image_grid(f"image_prediction", image_prediction)
        self.log(f"loss",loss)

        return loss

    def blend_fixed_amount_of_noise_with_each_sample(self,batch):
        p = self.hparams

        noise = torch.randn_like(batch)

        b,c,h,w = batch.shape

        r = torch.ones((b,1,1,1),device=self.device)*p.ratio_of_noise

        noisy_batch = torch.sqrt(1-r) * batch + torch.sqrt(r)*noise

        return noisy_batch

    def validation_step(self, batch, batch_idx):
        image = batch["image"]
        image_index = batch["index"]

        image_noisy = self.blend_fixed_amount_of_noise_with_each_sample(image)

        image_prediction = self.model(image_noisy)

        difficulty_loss = self.compute_difficulty_loss(image_prediction, image)

        return {"index":image_index,"loss":difficulty_loss}

    def compute_difficulty_loss(self,predicted,target):
        loss = torch.abs(predicted-target)
        loss = loss.mean(dim=(1,2,3))
        return loss

    def validation_epoch_end(self,validation_step_output_list):

        dict_of_validation_tensors = self.concat_validation_output(validation_step_output_list)
        
        image_index = dict_of_validation_tensors["index"]
        difficulty_loss = dict_of_validation_tensors["loss"]

        difficulty_index = self.compute_difficulty_index_for_each_loss(difficulty_loss)

        print(torch.bincount(difficulty_index))

    def concat_validation_output(self,validation_step_output_list):
        dict_of_lists = {}

        # Initialise empty lists for each output keys
        for key in validation_step_output_list[0].keys():
            dict_of_lists[key] = []

        # Append all the same outputs together
        for validation_step_output in validation_step_output_list:
            for key,value in validation_step_output.items():
                dict_of_lists[key].append(value)

        # Stack the lists of tensor together for each key
        for key in validation_step_output_list[0].keys():
            dict_of_lists[key] = torch.concat(dict_of_lists[key])

        return dict_of_lists

    def compute_difficulty_index_for_each_loss(self,loss):
        p = self.hparams

        loss_min = loss.min()
        loss_max = loss.max()

        loss_normalised = (loss-loss_min)/(loss_max-loss_min)

        loss_normalised = loss_normalised.clamp(0,0.99999)

        difficulty_index = (loss_normalised * p.number_of_classes).long()
        
        return difficulty_index

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


    