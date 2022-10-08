import torch
from torch.utils.data import Dataset
from pathlib import Path
from itertools import chain
import torchvision.io 


class ImageDataset(Dataset):
    
    def __init__(self,image_list_path, transform=None):
        self.image_list_path = Path(image_list_path)
        self.transform = transform
        self.image_path_list = self.read_list_of_image_paths()

    def read_list_of_image_paths(self):
        image_path_list = []

        with open(self.image_list_path) as f:

            lines = f.readlines()

            for relative_image_path in lines:
                relative_image_path = relative_image_path.strip()
                path = self.image_list_path.parent / relative_image_path
                image_path_list.append(path)

        return image_path_list
    

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self,index):

        image_path = self.image_path_list[index]
        image_path = str(image_path.resolve())

        image = torchvision.io.read_image(image_path).float()

        if self.transform is not None:
            image = self.transform(image)

        return {"image":image, "index":index}
        