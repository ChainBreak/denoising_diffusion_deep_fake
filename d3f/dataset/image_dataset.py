import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path
from itertools import chain

import albumentations as A



class ImageDataset(Dataset):
    
    def __init__(self,root_path, albumentations_transform=None):
        self.root_path = Path(root_path)
        self.albumentations_transform = albumentations_transform
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

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.albumentations_transform is not None:
            image = self.albumentations_transform(image=image)

        return image
        