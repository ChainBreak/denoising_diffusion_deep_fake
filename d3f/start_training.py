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

    def create_model_instance(self):
        p = self.hparams

        number_of_classes = 2
        output_size = 2 + number_of_classes

        model = segmentation_models_pytorch.Unet(
            encoder_name=p.encoder_name,
            encoder_weights=None,
            in_channels=3,
            classes=output_size,
            activation=None,
        )

        return model

    def configure_optimizers(self):
        p = self.hparams
        optimizer = torch.optim.Adam(self.model.parameters(), lr=p.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):

        p = self.hparams

        real_a = batch["a"]["raw_image"]
        real_b = batch["b"]["raw_image"]
        augm_a = batch["a"]["augmented_image"]
        augm_b = batch["b"]["augmented_image"]

        fake_a, fake_b = self.create_fakes_via_gradient_decent(real_a,real_b)

        augm_a_target = self.create_target_tensor_like(augm_a, is_fake=None, class_index=0)
        augm_b_target = self.create_target_tensor_like(augm_b, is_fake=None, class_index=1)
        real_a_target = self.create_target_tensor_like(real_a, is_fake=False, class_index=None)
        real_b_target = self.create_target_tensor_like(real_b, is_fake=False, class_index=None)
        fake_a_target = self.create_target_tensor_like(fake_a, is_fake=True , class_index=None)
        fake_b_target = self.create_target_tensor_like(fake_b, is_fake=True , class_index=None)

        self.log_batch_as_image_grid(f"real/a", real_a)
        self.log_batch_as_image_grid(f"real/b", real_b)
        self.log_batch_as_image_grid(f"augm/a", augm_a)
        self.log_batch_as_image_grid(f"augm/b", augm_b)
        self.log_batch_as_image_grid(f"fake/a", fake_a)
        self.log_batch_as_image_grid(f"fake/b", fake_b)
        

        input_tensor = torch.concat(
            (augm_a, augm_b, real_a, real_b, fake_a, fake_b),
            dim=0,
        )

        target_tensor = torch.concat(
            (augm_a_target, augm_b_target, real_a_target, real_b_target, fake_a_target, fake_b_target),
            dim=0,
        )

        output_tensor = self.model(input_tensor)

        loss_is_fake, loss_class = self.descriminator_loss(output_tensor,target_tensor)

        self.log("loss_is_fake",loss_is_fake)
        self.log("loss_class",loss_class)
        
        return loss_is_fake + loss_class

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
        print()
        for i in range(p.number_of_fake_generation_steps):

            model_output = self.model(input_tensor)

            loss_is_fake, loss_class = self.descriminator_loss(model_output, target_tensor)


            loss = loss_is_fake + loss_class
            loss.backward()

            grad_abs = input_tensor.grad.abs()
            grad_mean = grad_abs.mean()
            grad_max = grad_abs.max()
            grad_min = grad_abs.min()

            print(f"fake={loss_is_fake.item():8f} class={loss_class.item():8f} grad_min={grad_min.item():8f} grad_mean={grad_mean.item():8f}  grad_max={grad_max.item():8f}")
          

            with torch.no_grad():
                input_tensor -= p.fake_generation_step_size * input_tensor.grad
            
            input_tensor.grad.zero_()

        self.model.zero_grad()


        fake_b = input_tensor.detach()[:batch_size_a ]
        fake_a = input_tensor.detach()[ batch_size_a:]

        return fake_a, fake_b

    def create_target_tensor_like(self,batch, is_fake, class_index):
        b,c,h,w = batch.shape
        device = batch.device

        if is_fake is None:
            is_fake = -1

        if class_index is None:
            class_index = -1

        target_tensor = torch.tensor(
            [[is_fake ,class_index]],
            device=device,
        )
        target_tensor = target_tensor.reshape(1,2,1,1).repeat(b,1,h,w)

        return target_tensor # [b,2,h,w]

    def descriminator_loss(self, input_tensor , label_tensor ):
    
        is_fake_tensor = input_tensor[:,:2,:,:]
        class_tensor = input_tensor[:,2:,:,:]

        is_fake_labels = label_tensor[:,0,:,:]
        class_labels = label_tensor[:,1,:,:]

        loss_is_fake = F.cross_entropy(is_fake_tensor, is_fake_labels, ignore_index=-1)
        loss_class = F.cross_entropy(class_tensor, class_labels, ignore_index=-1)

        return loss_is_fake, loss_class
        
    def log_batch_as_image_grid(self,tag, batch):

        if self.global_step % 10 == 0:
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