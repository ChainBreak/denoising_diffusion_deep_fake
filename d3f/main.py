import click
from .train_deep_fake.start_training import train
from .balance_training_images.balance_training_images import balance
from .train_denoiser.train_denoiser import denoise

@click.group()
def cli():
    pass

cli.add_command(train)
cli.add_command(balance)
cli.add_command(denoise)

if __name__ == "__main__":
    cli()