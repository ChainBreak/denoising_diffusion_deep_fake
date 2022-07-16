import d3f
import math
import yaml
import argparse
import pytorch_lightning as pl
import segmentation_models_pytorch
import torch
from torch import sqrt
import torch.nn as nn
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

        model = segmentation_models_pytorch.FPN(
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

        if batch_idx == 0 :
            self.log_tensor_as_image("dataset/a", batch_a)
            self.log_tensor_as_image("dataset/b", batch_b)

        if optimizer_idx == 0:
            self.training_step_for_one_model( self.model_a, batch_a, self.model_b)
            
        if optimizer_idx == 1:
            self.training_step_for_one_model( self.model_b, batch_b, self.model_a)
            
        return 


    def training_step_for_one_model(self, this_model, this_batch, other_model):
        
        this_noisy_batch = self.blend_images_with_noise(this_batch)

        self.log_tensor_as_image("this_noise_batch/a", this_noisy_batch)

        other_fake_batch = self.iteratively_denoise_image(other_model, this_noisy_batch)

        self.log_tensor_as_image("other_fake_batch/a", other_fake_batch)

        target_noise = self.blend_images_with_noise(other_fake_batch)

        self.log_tensor_as_image("target_noise/a", target_noise)

        # input_batch, times = self.create_diffusion_samples(this_batch, target_noise)

        # noise_prediction = this_model(input_batch)

        # loss = self.mse_loss(noise_prediction, target_noise)

        return loss

    def blend_images_with_noise(self,batch):
        p = self.hparams

        r = p.noise_blend_ratio

        noise = torch.randn_like(batch)

        noisy_batch = math.sqrt(1-r) * batch + math.sqrt(r) * noise

        return noisy_batch


    def iteratively_denoise_image(self,model, xt):
        
        with torch.no_grad():
            p = self.hparams

            steps = p.number_denoising_steps

            alpha_t_list = torch.linspace(1, 0, steps+1, device=self.device)

            for i in range(steps):

                alpha_t = alpha_t_list[i]
                next_alpha_t = alpha_t_list[1+1]

                noise_prediction = model(xt)

                x_scale = sqrt(next_alpha_t) / sqrt(alpha_t)

                noise_scale = ( sqrt(alpha_t) * sqrt(1-next_alpha_t) - sqrt(next_alpha_t) * sqrt(1-alpha_t) ) / sqrt(alpha_t)
                
                xt =  x_scale * xt + noise_scale * noise_prediction

                xt = xt.clamp(-1,1)

            return xt




    def log_tensor_as_image(self,tag, batch):
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