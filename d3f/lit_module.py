import cv2
import numpy as np
import math
import argparse
import random
import pytorch_lightning as pl
import segmentation_models_pytorch
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torchvision
from torch.utils.data import DataLoader
from d3f.dataset.image_dataset import ImageDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from kornia import augmentation as K
from kornia.augmentation import AugmentationSequential



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

        transform = self.create_dataloader_augmentation_pipeline(mean,std)

        dataset = ImageDataset(
            path,
            albumentations_transform=transform,
            )

        dataloader = DataLoader(
            dataset=dataset, 
            batch_size=p.batch_size,
            num_workers=p.num_workers,
            shuffle=True,
            )
        
        return dataloader

    def create_dataloader_augmentation_pipeline(self,mean,std):
         return A.Compose([
            A.Normalize(mean,std,max_pixel_value=1.0),
            ToTensorV2(),
        ])

    def configure_optimizers(self):
        p = self.hparams
        optimizer_a = torch.optim.Adam(self.model_a.parameters(), lr=p.learning_rate)
        optimizer_b = torch.optim.Adam(self.model_b.parameters(), lr=p.learning_rate)
        return [optimizer_a, optimizer_b]

    def training_step(self, batch, batch_idx, optimizer_idx):
        
        batch_a = batch["a"]["image"]
        batch_b = batch["b"]["image"]

        if optimizer_idx == 0:
            loss = self.training_step_for_one_model("a", batch_a, self.model_a, self.model_b)
            
        if optimizer_idx == 1:
            loss = self.training_step_for_one_model("b", batch_b, self.model_b, self.model_a)
            
        return loss

    def training_step_for_one_model(self,name, real, real_model, fake_model):
        
        with torch.no_grad():
            fake = fake_model(real) 

        aug_real, aug_fake = self.apply_the_same_augmentation_to_list_of_image_tensors([real,fake])

        real_prediction = real_model(aug_fake)
        
        loss = self.mse_loss(real_prediction, aug_real)

        self.log_batch_as_image_grid(f"fake/{name}_to_fake", fake)
        self.log_batch_as_image_grid(f"real/{name}", real)
        self.log_batch_as_image_grid(f"model_input/{name}", aug_fake)
        self.log_batch_as_image_grid(f"model_target/{name}", aug_real)
        self.log_batch_as_image_grid(f"model_prediction/{name}", real_prediction)
        self.log(f"loss/train_{name}",loss)

        return loss


    def apply_the_same_augmentation_to_list_of_image_tensors(self,image_tensor_list):
        
        random_seed = random.randint(0,2**32)

        augmented_tensor_list = []

        for image_tensor in image_tensor_list:

            torch.manual_seed(random_seed)

            aug_list = AugmentationSequential(
                K.RandomAffine(
                    degrees=10, 
                    translate=[0.1, 0.1], 
                    scale=[0.75, 1.25], 
                    shear=10, 
                    p=0.5),
            )
            augmented_image_tensor = aug_list(image_tensor)

            augmented_tensor_list.append(augmented_image_tensor)


        return augmented_tensor_list
        
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
        