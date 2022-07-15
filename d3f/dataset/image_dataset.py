import torch
from torch.utils.data import Dataset
from pathlib import Path
from itertools import chain
import torchvision.io 


class ImageDataset(Dataset):
    
    def __init__(self,root_path, transform=None):
        self.root_path = Path(root_path)
        self.transform = transform
        self.image_path_list = self.get_list_of_image_paths()

    def get_list_of_image_paths(self):
        extensions = ["*.jpg","*.png"] 
        paths = [ self.root_path.rglob(extension) for extension in extensions]
        return list(chain(*paths))

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self,index):

        image_path = self.image_path_list[index]
        image_path = str(image_path.resolve())

        image = torchvision.io.read_image(image_path).float()

        if self.transform is not None:
            image = self.transform(image)

        return image
        