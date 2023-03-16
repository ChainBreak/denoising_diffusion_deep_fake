import click
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from d3f.train_deep_fake.lit_module import LitModule
from datetime import timedelta

@click.command()
@click.option("--config_path", required=True, help="Path to the config yaml.",)
def train(config_path):
    """Train the deep fake"""

    hparams_dict = read_yaml_file_into_dict(config_path)

    start_training(hparams_dict)


def read_yaml_file_into_dict(yaml_file_path):
    
    with open(yaml_file_path) as f:
        return yaml.safe_load(f)

def start_training(hparams_dict):

    lit_module = LitModule(**hparams_dict)

    p = lit_module.hparams

    callback_list = [
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(
            save_last=True,
            save_top_k=-1,
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

if __name__ == "__main__":
    train()


