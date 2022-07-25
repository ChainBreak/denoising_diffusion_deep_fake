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

TARGET_A = 0
TARGET_B = 1
TARGET_REAL = 0
TARGET_FAKE = 1


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
            
        self.generator_a = self.create_generator_model_instance()
        self.generator_b = self.create_generator_model_instance()
        self.discriminator = self.create_discriminator_model_instance()

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

    def create_generator_model_instance(self):
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

    def create_discriminator_model_instance(self):

        model = torchvision.models.resnet18(
            num_classes=4,
        )

        return model

    def configure_optimizers(self):
        p = self.hparams
        optimizer_a = torch.optim.SGD(self.generator_a.parameters(), lr=p.generator_learning_rate)
        optimizer_b = torch.optim.SGD(self.generator_b.parameters(), lr=p.generator_learning_rate)
        optimizer_d = torch.optim.SGD(self.discriminator.parameters(), lr=p.discriminator_learning_rate)
        return [optimizer_a, optimizer_b, optimizer_d]

    def training_step(self, batch, batch_idx, optimizer_idx):
        p = self.hparams

        real_a = batch["a"]
        real_b = batch["b"]
    
        if optimizer_idx == 0:
        
            loss, image_b_to_a = self.generator_training_step(
                name="a", 
                starting_image=real_b, 
                generator=self.generator_a,
                target=self.create_target_tensor_like(real_b, TARGET_REAL, TARGET_A)
                )
            batch["b_to_a"] = image_b_to_a.detach()
            return loss
            
        if optimizer_idx == 1:
            loss, image_a_to_b = self.generator_training_step(
                name="b", 
                starting_image=real_a, 
                generator=self.generator_b,
                target=self.create_target_tensor_like(real_a, TARGET_REAL, TARGET_B)
                )
            batch["a_to_b"] = image_a_to_b.detach()
            return loss

        if optimizer_idx == 2:

            image = torch.concat([
                batch["a"],
                batch["b"],
                batch["a_to_b"],
                batch["b_to_a"],
            ],dim=0) 

            target = torch.concat([
                self.create_target_tensor_like( batch["a"], TARGET_REAL, TARGET_A),
                self.create_target_tensor_like( batch["b"], TARGET_REAL, TARGET_B),
                self.create_target_tensor_like( batch["a_to_b"], TARGET_FAKE, TARGET_A),
                self.create_target_tensor_like( batch["b_to_a"], TARGET_FAKE, TARGET_B),
            ],dim=0)

            prediction = self.discriminator( image )
            
            loss_real_or_fake, loss_a_or_b = self.discriminator_loss( prediction, target)
            loss = loss_real_or_fake + loss_a_or_b

            self.log(f"loss_real_or_fake/discriminator", loss_real_or_fake)
            self.log(f"loss_a_or_b/discriminator", loss_a_or_b)
            self.log(f"loss/discriminator", loss)

            return loss

            
    def generator_training_step(self, name, starting_image, generator, target):
 
        schedule = self.get_training_schedule_dict()

        fake_image = self.iteratively_generate_fake_image_from_real(
            start_image=starting_image,
            generator=generator,
            number_of_steps=schedule["number_of_generation_steps"],
        )
        fake_image = generator( fake_image )
        prediction = self.discriminator( fake_image )

        loss_real_or_fake, loss_a_or_b = self.discriminator_loss( prediction, target)
        loss = loss_real_or_fake + loss_a_or_b

        self.log(f"loss_real_or_fake/generator_{name}", loss_real_or_fake)
        self.log(f"loss_a_or_b/generator_{name}", loss_a_or_b)
        self.log(f"loss/generator_{name}", loss)
        self.log_batch_as_image_grid(f"starting_image/generator_{name}", starting_image)
        self.log_batch_as_image_grid(f"fake_image/generator_{name}", fake_image)

        return loss, fake_image


    def iteratively_generate_fake_image_from_real(self, start_image, generator, number_of_steps):
        a = 0.05
                
        with torch.no_grad():

            fake_image = start_image

            for step_i in range(number_of_steps):

                next_fake_image = generator(fake_image)

                fake_image = math.sqrt(1-a) * fake_image + math.sqrt(a) * next_fake_image

            return fake_image


    def discriminator_loss(self, input_tensor , target_tensor ):
    
        real_or_fake_tensor = input_tensor[:,:2]
        a_or_b_tensor = input_tensor[:,2:]

        real_or_fake_target = target_tensor[:,0]
        a_or_b_target = target_tensor[:,1]

        loss_real_or_fake = F.cross_entropy(real_or_fake_tensor, real_or_fake_target, ignore_index=-1)
        loss_a_or_b = F.cross_entropy(a_or_b_tensor, a_or_b_target, ignore_index=-1)

        return loss_real_or_fake, loss_a_or_b


    def create_target_tensor_like(self,batch, real_or_fake, a_or_b):
        b,c,h,w = batch.shape
        device = batch.device

        if real_or_fake is None:
            real_or_fake = -1

        if a_or_b is None:
            a_or_b = -1

        target_tensor = torch.tensor(
            [[real_or_fake, a_or_b]],
            device=device,
        )
        target_tensor = target_tensor.reshape(1,2).repeat(b,1)

        return target_tensor # [b,2]


    def get_training_schedule_dict(self):

        step = self.global_step

        schedule_dict = {
            "starting_noise_ratio": self.linear_interpolation(
                x=step,
                x1=0, x2=10000,
                y1=1.0, y2=0.0,
            ),
            "number_of_generation_steps": int(self.linear_interpolation(
                x=step,
                x1=0, x2=1000,
                y1=0, y2=20,
            )),

        }

        self.log_dict(schedule_dict)

        return schedule_dict

    @staticmethod
    def linear_interpolation(x,x1,x2,y1,y2):
        
        y = (x-x1) / (x2-x1) * (y2-y1) + y1

        y_min = min(y1,y2)
        y_max = max(y1,y2)

        y = min(y_max,max(y_min,y))

        return y

      
    def log_batch_as_image_grid(self,tag, batch):

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
        

if __name__ == "__main__":
    main()