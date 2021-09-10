from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.models import resnet50
from torchvision.io import read_image


class FaceClassifier(nn.Module):
    def __init__(self, pretrained=True):
        super.__init__()
        self.backbone = resnet50(pretrained=pretrained)
        self.fc = nn.Linear(in_features=-1, out_features=2)

    def forward(self, x):
        x = self.backbone(x)
        x = F.relu(x)
        x = self.fc(x)
        x = F.softmax(x, dim=-1)
        return x


class ImageDataDataset(Dataset):
    def __init__(self, imgP_dir, imgN_dir, transform=None, target_transform=None):
        imgPs = list(Path(imgP_dir).glob('*'))
        imgNs = list(Path(imgN_dir).glob('*'))
        self.imgPaths = imgPs + imgNs
        self.label = torch.cat(
            (torch.ones(len(imgPs)), torch.zeros(len(imgPs))))

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        img_path = self.imgPaths[idx]
        label = self.label[idx]
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform
        return image, label
