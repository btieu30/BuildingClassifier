import os
import pandas as pd
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset

class BuildingDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):

        # changing Professor Rado's code a bit here because it was not checking for file existence
        df = pd.read_csv(annotations_file)

        valid_indices = []
        for idx in range(len(df)):
            img_name = df.iloc[idx, 0]
            if os.path.exists(os.path.join(img_dir, img_name)):
                valid_indices.append(idx)

        self.img_labels = df.iloc[valid_indices].reset_index(drop=True)

        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)

        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
