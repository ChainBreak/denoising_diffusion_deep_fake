import d3f
import yaml
import argparse
import pytorch_lightning as pl
import segmentation_models_pytorch


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


class LitTrainer(pl.LightningModule):
    def __init__(self,**kwargs):
        super().__init__()
        
        self.save_hyperparameters()
            
        self.model = self.create_model_instance()

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


if __name__ == "__main__":
    main()
