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
        )

    trainer.fit(
        model=lit_trainer,
        )


class LitTrainer(pl.LightningModule):
    def __init__(self,**kwargs):
        super().__init__()
        
        self.save_hyperparameters()
            
        self.model = self.create_model_instance()

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
        number_of_classes = 2
        output_size = 2 + number_of_classes

        model = torchvision.models.resnet34(
            num_classes=output_size,
        )

        return model

    def configure_optimizers(self):
        p = self.hparams
        optimizer = torch.optim.Adam(self.model.parameters(), lr=p.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        self.current_batch = batch_idx
        p = self.hparams

        real_a = batch["a"]
        real_b = batch["b"]

        fake_a, fake_b = self.create_fakes_via_gradient_decent(real_a,real_b)


        real_a_target = self.create_target_tensor_like(real_a, is_fake=False, class_index=0)
        real_b_target = self.create_target_tensor_like(real_b, is_fake=False, class_index=1)
        fake_a_target = self.create_target_tensor_like(fake_a, is_fake=True , class_index=0)
        fake_b_target = self.create_target_tensor_like(fake_b, is_fake=True , class_index=1)

        self.log_batch_as_image_grid(f"real_a", real_a, first_batch_only=True)
        self.log_batch_as_image_grid(f"real_b", real_b, first_batch_only=True)
        self.log_batch_as_image_grid(f"fake_a", fake_a, first_batch_only=True)
        self.log_batch_as_image_grid(f"fake_b", fake_b, first_batch_only=True)
        

        input_tensor = torch.concat(
            (real_a, real_b, fake_a, fake_b),
            dim=0,
        )

        target_tensor = torch.concat(
            (real_a_target, real_b_target, fake_a_target, fake_b_target),
            dim=0,
        )

        output_tensor = self.model(input_tensor)

        loss = self.descriminator_loss(output_tensor,target_tensor)

        self.log("loss",loss)
        
        return loss

    def create_fakes_via_gradient_decent(self,real_a,real_b):
        p = self.hparams

        batch_size_a = real_a.shape[0]
        batch_size_b = real_b.shape[0]

        # Targets are a real images with the other items class
        target_a = self.create_target_tensor_like(real_a, is_fake=False, class_index=1)
        target_b = self.create_target_tensor_like(real_b, is_fake=False, class_index=0)

        input_tensor = torch.concat(
            (real_a, real_b),
            dim=0,
        )

        target_tensor = torch.concat(
            (target_a, target_b),
            dim=0,
        )

        input_tensor.requires_grad_()

        for i in range(p.number_of_fake_generation_steps):

            model_output = self.model(input_tensor)

            loss = self.descriminator_loss(model_output, target_tensor)
    
            loss.backward()

            with torch.no_grad():
                input_tensor += p.fake_generation_step_size * input_tensor.grad
            
            input_tensor.grad.zero_()


        fake_b = input_tensor.detach()[:batch_size_a ]
        fake_a = input_tensor.detach()[ batch_size_a:]

        return fake_a, fake_b



    def create_target_tensor_like(self,batch, is_fake, class_index):
        b = batch.shape[0]
        device = batch.device

        target_tensor = torch.tensor(
            [[is_fake ,class_index]],
            device=device,
        )
        target_tensor = target_tensor.repeat(b,1)

        return target_tensor # [b,2]

    def descriminator_loss(self, input_tensor , label_tensor ):
    
        is_fake_tensor = input_tensor[:,:2]
        class_tensor = input_tensor[:,2:]

        is_fake_labels = label_tensor[:,0]
        class_labels = label_tensor[:,1]

        loss_is_fake = F.cross_entropy(is_fake_tensor, is_fake_labels)
        loss_class = F.cross_entropy(class_tensor, class_labels)

        loss = loss_is_fake + loss_class

        return loss


    def training_step_for_one_model(self,name, this_model, this_batch, other_model,):
        b,c,h,w = this_batch.shape

        steps = 10

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

        if first_batch_only and self.current_batch > 0:
            return

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