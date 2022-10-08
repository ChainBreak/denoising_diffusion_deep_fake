import click
from .train_deep_fake.start_training import train
from .balance_training_images.balance_training_images import balance

@click.group()
def cli():
    pass

cli.add_command(train)
cli.add_command(balance)

if __name__ == "__main__":
    cli()