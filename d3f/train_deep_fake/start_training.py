import click
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from d3f.train_deep_fake.lit_module import LitModule
from datetime import timedelta

@click.group()
def train():
    pass

@train.command()
@click.option("--config_path", required=True, help="Path to the config yaml.")
def new(config_path):
    hparams_dict = read_yaml_file_into_dict(config_path)
    lit_module = LitModule(**hparams_dict)
    start_training(lit_module)

@train.command()
@click.option("--checkpoint_path", required=True, help="Path to model checkpoint.")
def resume(checkpoint_path):
    lit_module = LitModule.load_from_checkpoint(checkpoint_path)
    start_training(lit_module)

@train.command()
@click.option("--config_path", required=True, help="Path to the config yaml.")
@click.option("--checkpoint_path", required=True, help="Path to model checkpoint.")
def modify(config_path, checkpoint_path):
    hparams_dict = read_yaml_file_into_dict(config_path)
    lit_module = LitModule.load_from_checkpoint(checkpoint_path,**hparams_dict)
    start_training(lit_module)

def read_yaml_file_into_dict(yaml_file_path):
    with open(yaml_file_path) as f:
        return yaml.safe_load(f)

def start_training(lit_module):
    p = lit_module.hparams

    print_hparams(p)
    
    callback_list = [
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(
            save_last=True,
            save_top_k=8,
            monitor="epoch",
            train_time_interval=timedelta(hours=2),
        ),
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

def print_hparams(p):
    print()
    print("Hyper Parameters:")
    for k,v in p.items():
        print(f"\t{k}: {v}")
    print()

if __name__ == "__main__":
    train()


