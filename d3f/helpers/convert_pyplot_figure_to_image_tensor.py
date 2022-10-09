
import io
import matplotlib.pyplot as plt
import PIL.Image
from torchvision.transforms import ToTensor
import torchvision.io 


def convert_pyplot_figure_to_image_tensor(figure):
    file_buffer = io.BytesIO()
    figure.savefig(file_buffer, format='jpeg')

    file_buffer.seek(0)

    image = PIL.Image.open(file_buffer)
    image = ToTensor()(image)

    return image