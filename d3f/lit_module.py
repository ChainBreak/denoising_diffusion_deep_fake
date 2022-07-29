import math
import argparse
import pytorch_lightning as pl
import segmentation_models_pytorch
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision
from torch.utils.data import DataLoader
from d3f.dataset.image_dataset import ImageDataset


class LitModule(pl.LightningModule):
    def __init__(self,**kwargs):
        super().__init__()
        
        self.save_hyperparameters()
            
        self.model_a = self.create_model_instance()
        self.model_b = self.create_model_instance()

        self.mse_loss = nn.MSELoss()

        self.current_batch = 0

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

    def configure_optimizers(self):
        p = self.hparams
        optimizer_a = torch.optim.Adam(self.model_a.parameters(), lr=p.learning_rate)
        optimizer_b = torch.optim.Adam(self.model_b.parameters(), lr=p.learning_rate)
        return [optimizer_a, optimizer_b]

    def training_step(self, batch, batch_idx, optimizer_idx):
        
        batch_a = batch["a"]
        batch_b = batch["b"]

        if optimizer_idx == 0:
            loss = self.training_step_for_one_model("a", batch_a, self.model_a, self.model_b)
            
        if optimizer_idx == 1:
            loss = self.training_step_for_one_model("b", batch_b, self.model_b, self.model_a)
            
        return loss

    def training_step_for_one_model(self,name, real, real_model, fake_model):
        
        with torch.no_grad():
            fake = fake_model(real) 

        real_prediction = real_model(fake)

        loss = self.mse_loss(real_prediction, real)

        self.log_batch_as_image_grid(f"fake/{name}_to_fake", fake)
        self.log_batch_as_image_grid(f"real_target/{name}", real)
        self.log_batch_as_image_grid(f"real_prediction/{name}", real_prediction)
        self.log(f"loss/train_{name}",loss)

        return loss
        
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
        