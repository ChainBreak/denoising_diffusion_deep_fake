import d3f
import yaml
import argparse
import pytorch_lightning as pl
import segmentation_models_pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
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

    def training_step(self, batch, batch_idx, optimizer_idx):
        batch_a = batch["a"]
        batch_b = batch["b"]

        # print(f"{optimizer_idx=}",batch_a[0,0,0,0],batch_b[0,0,0,0])
        if optimizer_idx == 0:
            pred = self.model_a(batch_a)
        if optimizer_idx == 1:
            pred = self.model_b(batch_b)
        return pred.sum()

    def configure_optimizers(self):
        p = self.hparams
        optimizer_a = torch.optim.Adam(self.model_a.parameters(), lr=p.learning_rate)
        optimizer_b = torch.optim.Adam(self.model_b.parameters(), lr=p.learning_rate)
        return [optimizer_a, optimizer_b]


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