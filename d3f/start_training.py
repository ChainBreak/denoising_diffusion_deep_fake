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
import albumentations as A
from d3f.dataset.image_dataset import ImageDataset

REAL_SIGN = -1
FAKE_SIGN = 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path",help="path to the config yaml")
    args = parser.parse_args()
    hparams_dict = read_yaml_file_into_dict(args.config_path)
    start_training(hparams_dict)

def read_yaml_file_into_dict(yaml_file_path):
    with open(yaml_file_path) as f:
        return yaml.safe_load(f)

def start_training(hparams_dict):
    lit_trainer = LitTrainer(**hparams_dict)
    trainer = pl.Trainer(
        gpus=1,
        log_every_n_steps=1,
        )

    trainer.fit(
        model=lit_trainer,
        )


class LitTrainer(pl.LightningModule):
    def __init__(self,**kwargs):
        super().__init__()
        
        self.save_hyperparameters()
            
        self.discriminator = self.create_discriminator_model_instance()

    def train_dataloader(self):
        p = self.hparams

        dataloader_a = self.create_dataloader(p.data_path_a, p.mean_a, p.mean_a)
        dataloader_b = self.create_dataloader(p.data_path_b, p.mean_b, p.mean_b)
  
        return {"a":dataloader_a, "b":dataloader_b}

    def create_dataloader(self, path, mean, std):
        p = self.hparams

        albumentations_transform = self.build_albumentations_transform_pipeline()

        dataset = ImageDataset(
            path,
            mean=mean,
            std=std,
            albumentations_transform = albumentations_transform,
            )

        dataloader = DataLoader(
            dataset=dataset, 
            batch_size=p.batch_size,
            num_workers=p.num_workers,
            shuffle=True,
            )
        
        return dataloader

    def build_albumentations_transform_pipeline(self):
        return A.Compose([
            # A.RandomRotate90(),
            # A.Flip(),
            # A.Transpose(),
            A.OneOf([
                # A.IAAAdditiveGaussianNoise(),
                A.GaussNoise(),
            ], p=0.2),
            A.OneOf([
                A.MotionBlur(p=.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.2, rotate_limit=10, p=0.2),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=.1),
                # A.IAAPiecewiseAffine(p=0.3),
            ], p=0.2),
            A.OneOf([
                A.CLAHE(clip_limit=2),

                A.RandomBrightnessContrast(),            
            ], p=0.3),
            A.HueSaturationValue(hue_shift_limit=255,p=1.0),
        ])

    def create_discriminator_model_instance(self):
        p = self.hparams

    
        model = torchvision.models.resnet18(
            num_classes=1,  
        )


        return model

    def configure_optimizers(self):
        p = self.hparams
        lr = p.learning_rate
        b1 = p.adam_b1
        b2 = p.adam_b2
        optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr,weight_decay=p.weight_decay)
        return optimizer

    def training_step(self, batch, batch_idx):

        # self.clamp_discriminator_paramters()

        driver = batch["a"]["augmented_image"]
        real = batch["b"]["augmented_image"]

        fake = self.create_fakes_via_gradient_decent(driver)

        real_loss  = REAL_SIGN * self.discriminator(real).mean()
        fake_loss  = FAKE_SIGN * self.discriminator(fake).mean()

        loss = real_loss + fake_loss


        self.log("loss/loss",loss)
        self.log("loss/real",real_loss)
        self.log("loss/fake",fake_loss)

        self.log_batch_as_image_grid(f"images/driver", driver)
        self.log_batch_as_image_grid(f"images/real", real)
        self.log_batch_as_image_grid(f"images/fake", fake)

            
        return loss

    def create_fakes_via_gradient_decent(self,driver):
        p = self.hparams

        fake = driver.clone()

        # Targets are a real images with the other items class
        target = self.create_target_tensor_like(driver, REAL_SIGN)

        fake.requires_grad_()
        
        number_of_steps = 1#self.global_step//100 + 1

        for i in range(random.randint(1,number_of_steps)):

            
            

            loss = REAL_SIGN*self.discriminator(fake).mean()

            loss.backward()

            grad_abs = fake.grad.abs()
            grad_mean = grad_abs.mean()
            grad_max = grad_abs.max()
            grad_min = grad_abs.min()

            print(f"loss={loss.item():8f}  grad_min={grad_min.item():8f} grad_mean={grad_mean.item():8f}  grad_max={grad_max.item():8f}")
          

            with torch.no_grad():
                fake -= p.fake_generation_step_size * fake.grad
            
            if i == 0:
                self.log_batch_as_image_grid(f"images/fake_grad", fake.grad / grad_mean)

            fake.grad.zero_()
            
        self.discriminator.zero_grad()

        self.log("loss_generator",loss)


        return fake

    def create_target_tensor_like(self,batch, real_or_fake):
        b,c,h,w = batch.shape
        device = batch.device

        target_tensor = torch.tensor(
            [real_or_fake],
            device=device,
        )
        target_tensor = target_tensor.reshape(1,1,1,1).repeat(b,1,h,w)

        return target_tensor # [b,1,h,w]

    def descriminator_loss(self, input_tensor , label_tensor ):
    
        loss = input_tensor * label_tensor

        return loss.mean()

    def clamp_discriminator_paramters(self):
        p = self.hparams
        clip_value = p.discriminator_parameter_clamp_value
        for p in self.discriminator.parameters():
            p.data.clamp_(-clip_value, clip_value)
        
    def log_batch_as_image_grid(self,tag, batch):

        if self.global_step % 100 == 0:
            nrows = 3
            ncols = 3
            n = nrows*ncols

            image = torchvision.utils.make_grid(batch[:n], nrows)

            image *= 0.5
            image += 0.5
            image = image.clamp(0,1)

            self.logger.experiment.add_image( tag, image, self.global_step)
        

if __name__ == "__main__":
    main()


# batch_a = dataloader_a.next()
# batch_a_to_b = model_b.denoise(batch_a)
# batch_a, batch_a_noise = add_noise(t, batch_a, batch_a_to_b)
# batch_a_pred = model_a(batch_a)
# loss_a = mse(batch_a_pred, batch_a_noise)

# batch_b = dataloader_b.next()
# batch_b_to_a = model_a.denoise(batch_b)
# batch_b, batch_b_noise = add_noise(t, batch_b, batch_b_to_a)
# batch_b_pred = model_b(batch_b)
# loss_b = mse(batch_b_pred, batch_b_noise)