import cv2
import numpy as np
import math
import argparse
from datetime import timedelta
import random
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
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

import albumentations as A
from albumentations.pytorch import ToTensorV2

from d3f.loss_functions import MseStructuralSimilarityLoss
from d3f.helpers import LoggingScheduler

from d3f.train_denoiser.lit_module import LitModule as DenoisingModel

from ema_pytorch import EMA

class LitModule(pl.LightningModule):
    def __init__(self,**kwargs):
        super().__init__()
        
        self.save_hyperparameters()
        p = self.hparams
        self.model_a = self.create_model_instance()
        self.model_b = self.create_model_instance()

        self.ema_model_a = self.create_ema_model(self.model_a)
        self.ema_model_b = self.create_ema_model(self.model_b)

        self.criterion = MseStructuralSimilarityLoss(-1.0,1.0)

        self.image_logging_scheduler = LoggingScheduler()

    def load_denoising_model_from_checkpoint(self,checkpoint_path):
        return DenoisingModel.load_from_checkpoint(checkpoint_path).model

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

    def create_ema_model(self,model):
        p = self.hparams
        if p.mode == "swap":
            return EMA(
                model,
                beta = p.ema_beta,
                update_every = p.ema_update_every,
                include_online_model = False,
            )

    def train_dataloader(self):
        p = self.hparams

        dataloader_a = self.create_dataloader(p.data_path_a, p.mean_a, p.mean_a)
        dataloader_b = self.create_dataloader(p.data_path_b, p.mean_b, p.mean_b)
  
        return {"a":dataloader_a, "b":dataloader_b}

    def create_dataloader(self, path, mean, std):
        p = self.hparams

        transform = self.create_augmentation_sequence(mean, std)

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

    def create_augmentation_sequence(self, mean, std):
        augmentation_sequence = A.Compose([
            A.Normalize(mean,std),
            A.ShiftScaleRotate(
                shift_limit=0.2,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=0,
                p=0.7,
            ),
            ToTensorV2(),
        ])
        return augmentation_sequence

    def configure_optimizers(self):
        p = self.hparams

        b1 = p.adam_b1
        b2 = p.adam_b2

        optimizer_a = optimizers.Adam(self.model_a.parameters(), lr=p.learning_rate,betas=(b1,b2))
        optimizer_b = optimizers.Adam(self.model_b.parameters(), lr=p.learning_rate,betas=(b1,b2))

        scheduler_a = schedulers.CosineAnnealingLR(optimizer_a, T_max=p.cosine_scheduler_max_epoch)
        scheduler_b = schedulers.CosineAnnealingLR(optimizer_b, T_max=p.cosine_scheduler_max_epoch)

        return [optimizer_a, optimizer_b], [scheduler_a, scheduler_b]

    def configure_callbacks(self):
        return [
            LearningRateMonitor(logging_interval='step'),
            ModelCheckpoint(
                save_top_k=8,
                monitor="epoch",
                mode="max",
                train_time_interval=timedelta(hours=2),
            ),
            ModelCheckpoint(
                filename="last",
                save_on_train_epoch_end=True,
            ),
        ]

    def training_step(self, batch, batch_idx, optimizer_idx):
        
        batch_a = batch["a"]["image"]
        batch_b = batch["b"]["image"]

        if optimizer_idx == 0:
            self.image_logging_scheduler.update_with_step_number(self.global_step)
            loss = self.training_step_for_one_model("a", batch_a, self.model_a, self.ema_model_b)
            
        if optimizer_idx == 1:
            loss = self.training_step_for_one_model("b", batch_b, self.model_b, self.ema_model_a)

        self.log("epoch",float(self.current_epoch))
            
        return loss

    def training_step_for_one_model(self, name, real, real_model, fake_model):
        p = self.hparams

        if p.mode == "denoise":
            loss = self.training_denoise_step_for_one_model(name, real, real_model)
        elif p.mode == "swap":
            loss = self.training_swap_step_for_one_model(name, real, real_model, fake_model)

        return loss

    def training_denoise_step_for_one_model(self, name, real, real_model):
        with torch.no_grad():
                   
            noisy_real = self.blend_random_amount_of_noise_with_each_sample(real)
    
        real_prediction = real_model(noisy_real)
        
        loss = self.criterion(real_prediction, real)

        self.log_batch_as_image_grid(f"denoise_1_model_input/{name}", noisy_real)
        self.log_batch_as_image_grid(f"denoise_2_model_prediction/{name}", real_prediction)
        self.log(f"loss_denoise/train_{name}",loss)

        return loss

    def training_swap_step_for_one_model(self, name, real, real_model, fake_model):

        fake_model.update()

        with torch.no_grad():
                   
            fake = fake_model(real) 

            swap_diff = nn.functional.mse_loss(real,fake)

            noisy_fake = self.blend_random_amount_of_noise_with_each_sample(fake)
    
        real_prediction = real_model(noisy_fake)
        
        loss = self.criterion(real_prediction, real)

        self.log_batch_as_image_grid(f"swap_1_real/{name}", real)
        self.log_batch_as_image_grid(f"swap_2_fake/{name}_to_fake",fake)
        self.log_batch_as_image_grid(f"swap_3_model_input/{name}", noisy_fake)
        self.log_batch_as_image_grid(f"swap_4_model_prediction/{name}", real_prediction)
        self.log(f"swap_difference/{name}",swap_diff)
        self.log(f"loss_swap/train_{name}",loss)

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

    def predict_fake(self,real_bgr,model_a_or_b):
        p = self.hparams
        if model_a_or_b == "a":
            return self.predict_fake_for_single_frame(real_bgr, self.model_a, p.mean_b, p.std_b)

        if model_a_or_b == "b":
            return self.predict_fake_for_single_frame(real_bgr, self.model_b, p.mean_a, p.std_a)

    def predict_fake_for_single_frame(self, real_bgr, model, mean ,std ):

        mean = torch.tensor(mean,device=self.device)
        std = torch.tensor(std,device=self.device)

        input_tensor = self.cv2_to_tensor_normalised(real_bgr, mean, std)

        output_tensor = model(input_tensor)

        fake_bgr = self.tensor_cv2_to_denormalised(output_tensor, mean, std)

        return fake_bgr

    def cv2_to_tensor_normalised(self,image_bgr,mean,std):

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
  
        tensor = torch.from_numpy(image_rgb).float().to(self.device)
 
        tensor = tensor.permute(2,0,1) # hwc to chw

        tensor -= mean.reshape(3,1,1)*255
        tensor /= std.reshape(3,1,1)*255

        return tensor.unsqueeze(0)

    def tensor_cv2_to_denormalised(self,tensor,mean,std):
        tensor = tensor.squeeze(0)

        tensor *= std.reshape(3,1,1)*255
        tensor += mean.reshape(3,1,1)*255

        tensor = tensor.permute(1,2,0) # chw to hwc

        tensor = tensor.int()
        tensor = tensor.clamp(0,255)

        image_rgb = tensor.cpu().numpy().astype(np.uint8)

        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        return image_bgr
        
