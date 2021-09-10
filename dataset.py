from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


class ImageDataDataset(Dataset):
    def __init__(self, imgP_dir, imgN_dir, transform=None, target_transform=None):
        imgPs = list(Path(imgP_dir).glob('*'))
        imgNs = list(Path(imgN_dir).glob('*'))
        self.imgPaths = imgPs + imgNs
        self.label = torch.cat(
            (torch.zeros(len(imgPs)), torch.ones(len(imgNs))))

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        img_path = self.imgPaths[idx]
        label = self.label[idx]
        image = read_image(img_path.as_posix())
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform
        return image, label
