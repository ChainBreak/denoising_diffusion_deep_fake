import click
import yaml
from .lit_module import LitModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor

@click.command()
@click.option("--config", 
    required=True, 
    help="Path to config yaml file",)
@click.option("--input_list", 
    required=True, 
    help="Path to text file that lists relative paths to each image",)
@click.option("--output_list", 
    required=True, 
    help="Path to text file that lists relative paths to each image",)
def balance( **options):
    """Assign difficulty class to each image for balanced sampling.
    
    This trains a model to denoise images. 
    It then bins the images based on the reconstuction loss.
    The bin indexes become the difficulty classes. 
    """
    print(options)

    hparams_dict = read_yaml_file_into_dict(options["config"])
    
    hparams_dict["input_image_list_path"] = options["input_list"]
    hparams_dict["output_image_list_path"] = options["output_list"]

    start_training(hparams_dict)

def read_yaml_file_into_dict(yaml_file_path):
    
    with open(yaml_file_path) as f:
        return yaml.safe_load(f)

def start_training(hparams_dict):

    lit_module = LitModule(**hparams_dict)

    p = lit_module.hparams

    callback_list = [
        LearningRateMonitor(logging_interval='step')
    ]

    trainer = pl.Trainer(
        gpus=1,
        log_every_n_steps=1,
        max_epochs=p.max_epochs,
        callbacks=callback_list,
        )

    trainer.fit(
        model=lit_module,
        )