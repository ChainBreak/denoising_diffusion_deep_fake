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
        self.current_batch = batch_idx
        p = self.hparams

        batch_a = batch["a"]
        batch_b = batch["b"]

    
        if optimizer_idx == 0:
            self.log_batch_as_image_grid("dataset/a", batch_a, first_batch_only=True)
            loss = self.training_step_for_one_model("a", self.model_a, batch_a, self.model_b)
            self.log("loss/train_a",loss)
            
        if optimizer_idx == 1:
            self.log_batch_as_image_grid("dataset/b", batch_b, first_batch_only=True)
            loss = self.training_step_for_one_model("b", self.model_b, batch_b, self.model_a)
            self.log("loss/train_b",loss)
            
        return loss


    def training_step_for_one_model(self,name, this_model, this_batch, other_model,):
        b,c,h,w = this_batch.shape

        steps = 20

        noise = torch.randn_like(this_batch)

        input_batch = self.randomly_interpolate_images(noise, this_batch)

        other_fake_batch = self.iteratively_remove_error(other_model, input_batch, steps)

        error_prediction = this_model(input_batch)
        target_batch = input_batch - this_batch
        loss = self.mse_loss(error_prediction, target_batch)

        self.log_batch_as_image_grid(f"other_fake_batch/{name}_to_other", other_fake_batch, first_batch_only=True)
        self.log_batch_as_image_grid(f"model_input/{name}", input_batch, first_batch_only=True)
        self.log_batch_as_image_grid(f"target_batch/{name}", target_batch, first_batch_only=True)
        self.log_batch_as_image_grid(f"error_prediction/{name}", error_prediction, first_batch_only=True)

        return loss

    def get_max_error_removal_steps_from_shedule(self):
        start_epoch = 0
        end_epoch = 50

        start_steps = 2
        end_steps = 20

        epoch = self.current_epoch

        steps = int((epoch-start_epoch) / (end_epoch-start_epoch) * (end_steps-start_steps) + start_steps)

        self.log("noise_removal_steps_schedule",steps)

        return steps

    def iteratively_remove_error(self,model, image, steps):
        
        with torch.no_grad():
            p = self.hparams

            image = image.clone()

            step_scale = p.error_step_scale

            for i in range(steps):

                error_prediction = model(image)

                image -= error_prediction * step_scale

                image = image.clamp(-1,1)

                # self.log_batch_as_image_grid(f"removal/{i}_to_other", image, first_batch_only=True)

            return image

    def randomly_interpolate_images(self, image1, image2):
        b,c,h,w = image1.shape

        alpha_t = torch.rand(
            size=(b,1,1,1),
            device=self.device,
        )

        image = sqrt(alpha_t) * image1 + sqrt(1-alpha_t)*image2

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

            self.logger.experiment.add_image( tag, image, self.current_epoch)
        

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