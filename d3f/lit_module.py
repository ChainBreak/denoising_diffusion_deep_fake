import d3f
import math
import yaml
import random
import argparse
import pytorch_lightning as pl
import segmentation_models_pytorch
import torch
from torch import sqrt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision
from d3f.dataset.image_dataset import ImageDataset


class LitModule(pl.LightningModule):
    def __init__(self,**kwargs):
        super().__init__()
        
        self.save_hyperparameters()
            
        self.model_a = self.create_model_instance()
        self.model_b = self.create_model_instance()

        self.mse_loss = nn.MSELoss()

        self.current_batch = 0

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
            batch_size=p.batch_size,
            num_workers=p.num_workers,
            shuffle=True,
            )
        
        return dataloader

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

    def configure_optimizers(self):
        p = self.hparams
        optimizer_a = torch.optim.Adam(self.model_a.parameters(), lr=p.learning_rate)
        optimizer_b = torch.optim.Adam(self.model_b.parameters(), lr=p.learning_rate)
        return [optimizer_a, optimizer_b]

    def training_step(self, batch, batch_idx, optimizer_idx):
        p = self.hparams

        batch_a = batch["a"]
        batch_b = batch["b"]

        if optimizer_idx == 0:
            loss = self.training_step_for_one_model("a", batch_a, self.model_a, self.model_b)
            
        if optimizer_idx == 1:
            loss = self.training_step_for_one_model("b", batch_b, self.model_b, self.model_a)
            
        return loss

    def training_step_for_one_model(self,name, real, real_model, fake_model):
        p = self.hparams

        fake = self.iteratively_step_towards_domain(
            fake_model, 
            real, 
            p.number_of_denoise_steps,
        )

        real_prediction = real_model(fake)

        loss = self.mse_loss(real_prediction, real)

        self.log_batch_as_image_grid(f"fake/{name}_to_fake", fake)
        self.log_batch_as_image_grid(f"real_target/{name}", real)
        self.log_batch_as_image_grid(f"real_prediction/{name}", real_prediction)
        self.log(f"loss/train_{name}",loss)

        return loss

    def get_training_schedule_dict(self):

        step = self.global_step

        schedule_dict = {
            "starting_noise_ratio": self.linear_interpolation(
                x=step,
                x1=0, x2=10000,
                y1=1.0, y2=0.0,
            ),
            "number_of_denoise_steps": int(self.linear_interpolation(
                x=step,
                x1=0, x2=10000,
                y1=0, y2=20,
            )),

        }

        return schedule_dict


    @staticmethod
    def linear_interpolation(x,x1,x2,y1,y2):
        
        y = (x-x1) / (x2-x1) * (y2-y1) + y1

        y_min = min(y1,y2)
        y_max = max(y1,y2)

        y = min(y_max,max(y_min,y))

        return y

    def iteratively_step_towards_domain(self,model, start_image, steps):
        p = self.hparams
        
        with torch.no_grad():
            image = model(start_image)
            # image = start_image.clone()

            # for a in torch.linspace(1,0,steps):

            #     image = math.sqrt(1-a)*model(image) + math.sqrt(a)*start_image

            return image

    def randomly_blend_image_batches(self, image_batch_list):

        image_batch_stack = torch.stack(image_batch_list,dim=1)
        b,n,c,h,w = image_batch_stack.shape
        
        dirichlet = torch.distributions.dirichlet.Dirichlet(
            concentration=torch.ones(b,n,device=self.device),
        )

        variance = dirichlet.sample().reshape(b,n,1,1,1)

        # sqrt ensures the variances stay the same
        std = variance.sqrt()

        image = (std * image_batch_stack).sum(1)

        return image

    def blend_image_batches(self, image1, image2, blend_ratio):
        # sqrt ensures the variances stay the same
        image = math.sqrt(1-blend_ratio)*image1 + math.sqrt(blend_ratio) * image2 

        return image
        
    def log_batch_as_image_grid(self,tag, batch, first_batch_only=False):

        p = self.hparams

        if self.global_step % p.log_images_every_n_steps == 0:

            nrows = 3
            ncols = 3
            n = nrows*ncols

            image = torchvision.utils.make_grid(batch[:n], nrows)

            image *= 0.5
            image += 0.5
            image = image.clamp(0,1)

            self.logger.experiment.add_image( tag, image, self.global_step)
        