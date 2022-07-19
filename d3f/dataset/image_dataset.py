import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path
from itertools import chain

import albumentations as A
from albumentations.pytorch import ToTensorV2


class ImageDataset(Dataset):
    
    def __init__(
        self,
        root_data_path,
        mean,
        std,
        albumentations_transform=None,
        ):

        self.root_data_path = Path(root_data_path)
        self.albumentations_transform = albumentations_transform
        self.image_path_list = self.get_list_of_image_paths()

        self.common_transform = A.Compose([
            A.Normalize(
                mean=mean,
                std=std,
            ),
            ToTensorV2(),
        ])

    def get_list_of_image_paths(self):
        extensions = ["*.jpg","*.png"] 
        paths = [ self.root_data_path.rglob(extension) for extension in extensions]
        return list(chain(*paths))

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self,index):

        image_path = self.image_path_list[index]
        image_path = str(image_path.resolve())

        raw_image = cv2.imread(image_path)
        raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

        

        if self.albumentations_transform:
            augmented_image = self.albumentations_transform(image=raw_image)
            augmented_image = self.common_transform(image=augmented_image["image"])
        else:
            augmented_image = self.common_transform(image=raw_image)

        raw_image = self.common_transform(image=raw_image)


        return {
            "raw_image": raw_image["image"],
            "augmented_image": augmented_image["image"],
        }
        