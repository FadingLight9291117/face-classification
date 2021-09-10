import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class FaceClassifier(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = resnet50(pretrained=pretrained)
        self.backbone.fc = nn.Linear(in_features=2048, out_features=2)

    def forward(self, x):
        x = self.backbone(x)
        x = F.softmax(x, dim=-1)
        return x


if __name__ == '__main__':
    model = FaceClassifier()
    print(model)
