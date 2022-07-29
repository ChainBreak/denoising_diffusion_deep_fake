
import yaml
import argparse
import pytorch_lightning as pl

from d3f.lit_module import LitModule


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
    lit_module = LitModule(**hparams_dict)
    trainer = pl.Trainer(
        gpus=1,
        log_every_n_steps=1,
        )

    trainer.fit(
        model=lit_module,
        )


if __name__ == "__main__":
    main()


