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

        model = torchvision.models.resnet18(
            num_classes=output_size,
        )

        return model

    def configure_optimizers(self):
        p = self.hparams
        optimizer = torch.optim.SGD(self.model.parameters(), lr=p.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        self.current_batch = batch_idx
        p = self.hparams

        real_a = batch["a"]
        real_b = batch["b"]

        a_to_b, b_to_a = self.create_fakes_via_gradient_decent(real_a,real_b)


        real_a_target = self.create_target_tensor_like(real_a, is_fake=False, class_index=0)
        real_b_target = self.create_target_tensor_like(real_b, is_fake=False, class_index=1)
        a_to_b_target = self.create_target_tensor_like(a_to_b, is_fake=True , class_index=0)
        b_to_a_target = self.create_target_tensor_like(b_to_a, is_fake=True , class_index=1)

        self.log_batch_as_image_grid(f"real/a", real_a, first_batch_only=True)
        self.log_batch_as_image_grid(f"real/b", real_b, first_batch_only=True)
        self.log_batch_as_image_grid(f"fake/a_to_b", a_to_b, first_batch_only=True)
        self.log_batch_as_image_grid(f"fake/b_to_a", b_to_a, first_batch_only=True)
        

        input_tensor = torch.concat(
            (real_a, real_b, a_to_b, b_to_a),
            dim=0,
        )

        target_tensor = torch.concat(
            (real_a_target, real_b_target, a_to_b_target, b_to_a_target),
            dim=0,
        )

        output_tensor = self.model(input_tensor)

        loss_is_fake, loss_class = self.descriminator_loss(output_tensor,target_tensor)

        loss = loss_is_fake + loss_class

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

            loss_is_fake, loss_class = self.descriminator_loss(model_output, target_tensor)

            r = i/p.number_of_fake_generation_steps

            loss = (r**2)*loss_is_fake + loss_class
    
            loss.backward()

            grad_abs = input_tensor.grad.abs()
            grad_mean = grad_abs.mean()
            grad_max = grad_abs.max()
            grad_min = grad_abs.min()

            print(f"loss_is_fake={loss_is_fake.item():8f} loss_class={loss_class.item():8f} grad_min={grad_min.item():8f} grad_mean={grad_mean.item():8f}  grad_max={grad_max.item():8f}")

            with torch.no_grad():
                input_tensor -= p.fake_generation_step_size * input_tensor.grad
            
            input_tensor.grad.zero_()

        self.model.zero_grad()


        a_to_b = input_tensor.detach()[:batch_size_a ]
        b_to_a = input_tensor.detach()[ batch_size_a:]

        return a_to_b, b_to_a

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

        

        return loss_is_fake,  loss_class
        
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