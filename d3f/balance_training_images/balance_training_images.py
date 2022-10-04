import click

@click.command()
@click.option("--input_list", 
    required=True, 
    help="Path to text file that lists relative paths to each image",)
@click.option("--output_list", 
    required=True, 
    help="Path to text file that lists relative paths to each image",)
@click.option("--num_classes", 
    default=10, 
    show_default=True,
    help="Number of classes",)
def balance( **options):
    """Assign difficulty class to each image for balanced sampling.
    
    This trains a model to denoise images. 
    It then bins the images based on the reconstuction loss.
    The bin indexes become the difficulty classes. 
    """
    print(options)