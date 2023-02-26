import cv2
import numpy as np
import math
from math import sqrt
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
from d3f.train_denoiser.lit_module import LitModule as DenoisingModel



class LitModule(pl.LightningModule):
    def __init__(self,**kwargs):
        super().__init__()
        
        self.save_hyperparameters()
        p = self.hparams
        
        self.denoising_model_a = self.load_denoising_model_from_checkpoint(p.denoising_model_a)
        self.denoising_model_b = self.load_denoising_model_from_checkpoint(p.denoising_model_b)

        self.model = self.create_model_instance()

        self.criterion = nn.MSELoss()

        self.shared_augmentation_sequence = self.create_shared_augmentation_sequence()

        self.image_logging_scheduler = LoggingScheduler()

    def load_denoising_model_from_checkpoint(self,checkpoint_path):
        return DenoisingModel.load_from_checkpoint(checkpoint_path)


    def create_model_instance(self):
        p = self.hparams
        encoder_name = p["encoder_name"]

        model = segmentation_models_pytorch.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=4,
            classes=3,
            activation=None,
        )
        return model

    def create_shared_augmentation_sequence(self):
        augmentation_sequence = AugmentationSequential(
            K.RandomAffine(
                degrees=10, 
                translate=[0.1, 0.1], 
                scale=[0.95, 1.05], 
                shear=0, 
                p=1.0,
            ),
        )
        return augmentation_sequence

    def train_dataloader(self):
        p = self.hparams

        dataloader_a = self.create_dataloader(p.data_path_a, p.mean_a, p.mean_a)
        dataloader_b = self.create_dataloader(p.data_path_b, p.mean_b, p.mean_b)
  
        return {"a":dataloader_a, "b":dataloader_b}

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
            batch_size=p.batch_size//2,
            num_workers=p.num_workers,
            shuffle=True,
            )
        
        return dataloader

    def configure_optimizers(self):
        p = self.hparams

        b1 = p.adam_b1
        b2 = p.adam_b2

        optimizer = optimizers.Adam(self.model.parameters(), lr=p.learning_rate,betas=(b1,b2))
 
        scheduler = schedulers.CosineAnnealingLR(optimizer, T_max=p.cosine_scheduler_max_epoch)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):

        self.image_logging_scheduler.update_with_step_number(self.global_step)
        
        batch_a = batch["a"]["image"]
        batch_b = batch["b"]["image"]

        real_a, fake_a = self.create_fake("a", batch_a, self.denoising_model_b)
        real_b, fake_b = self.create_fake("b", batch_b, self.denoising_model_a)
        
        real = torch.concat([real_a,real_b],dim=0)
        fake = torch.concat([fake_a,fake_b],dim=0)

        
        real_prediction = self.model(fake)
        
        loss = self.criterion(real_prediction, real)
        
        self.log_batch_as_image_grid(f"model_input", fake[:,0:3,:,:])
        self.log_batch_as_image_grid(f"model_target", real)
        self.log_batch_as_image_grid(f"model_prediction", real_prediction)
        self.log(f"loss/train",loss)
            
        return loss

    def create_fake(self,name, real, fake_denoising_model):
        
        with torch.no_grad():
            b,c,h,w = real.shape
            aug_real = self.shared_augmentation_sequence(real)

            if name == "a":
                indicator = torch.zeros((b,1,h,w),device=self.device)
            if name == "b":
                indicator = torch.ones((b,1,h,w),device=self.device)

            noisy_aug_real = self.add_scheduled_amount_of_noise(aug_real)
            noisy_aug_real_indicator = torch.concat((noisy_aug_real,indicator),dim=1)

            # Generate the best fake we can
            aug_fake = self.model(noisy_aug_real_indicator) 

            aug_denoised_fake = fake_denoising_model(aug_fake)

            if name == "b":
                indicator = torch.zeros((b,1,h,w),device=self.device)
            if name == "a":
                indicator = torch.ones((b,1,h,w),device=self.device)

            noisy_aug_denoised_fake = self.add_scheduled_amount_of_noise(aug_denoised_fake)
            aug_denoised_fake_indicator = torch.concat((noisy_aug_denoised_fake,indicator),dim=1)

        self.log_batch_as_image_grid(f"1_real/{name}", real)
        self.log_batch_as_image_grid(f"2_fake/{name}_to_fake", aug_fake)
        self.log_batch_as_image_grid(f"3_denoised_fake/{name}_to_fake", aug_denoised_fake[:,0:3,:,:])


        return aug_real, aug_denoised_fake_indicator

    def add_scheduled_amount_of_noise(self,image):
        p = self.hparams

        alpha = (self.current_epoch+2) / p.noise_scheduler_max_epoch
        alpha = min(alpha,1.0)

        noise = torch.randn_like(image)

        noisy_image = sqrt(alpha)*image + sqrt(1-alpha)*noise

        return noisy_image

    def apply_the_same_augmentation_to_list_of_image_tensors(self,image_tensor_list, augmentation_sequence):
        
        augmented_tensor_list = []

        augmentation_params = None

        for image_tensor in image_tensor_list:

            augmented_image_tensor = augmentation_sequence(
                image_tensor,
                params=augmentation_params)

            if augmentation_params == None:
                augmentation_params = augmentation_sequence._params

            augmented_tensor_list.append(augmented_image_tensor)

        return augmented_tensor_list
        
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

    def predict_fake(self,real_bgr,model_a_or_b):
        p = self.hparams
        if model_a_or_b == "a":
            return self.predict_fake_for_single_frame(real_bgr, self.model_a, self.denoising_model_a, p.mean_b, p.std_b)

        if model_a_or_b == "b":
            return self.predict_fake_for_single_frame(real_bgr, self.model_b, self.denoising_model_b, p.mean_a, p.std_a)

    def predict_fake_for_single_frame(self, real_bgr, fake_model, fake_denoising_model, mean ,std ):

        mean = torch.tensor(mean,device=self.device)
        std = torch.tensor(std,device=self.device)

        input_tensor = self.cv2_to_tensor_normalised(real_bgr, mean, std)

        output_tensor = fake_denoising_model(fake_model(input_tensor))

        fake_bgr = self.tensor_cv2_to_denormalised(output_tensor, mean, std)

        return fake_bgr

    def cv2_to_tensor_normalised(self,image_bgr,mean,std):

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
  
        tensor = torch.from_numpy(image_rgb).float().to(self.device)
 
        tensor = tensor.permute(2,0,1) # hwc to chw

        tensor -= mean.reshape(3,1,1)
        tensor /= std.reshape(3,1,1)

        return tensor.unsqueeze(0)

    def tensor_cv2_to_denormalised(self,tensor,mean,std):
        tensor = tensor.squeeze(0)

        tensor *= std.reshape(3,1,1)
        tensor += mean.reshape(3,1,1)

        tensor = tensor.permute(1,2,0) # chw to hwc

        tensor = tensor.int()
        tensor = tensor.clamp(0,255)

        image_rgb = tensor.cpu().numpy().astype(np.uint8)

        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        return image_bgr
        
