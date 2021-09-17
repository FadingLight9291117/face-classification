import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision.models.mobilenetv2 import mobilenet_v2


class FaceClassifier(nn.Module):
    def __init__(self, model_name='resnet50', pretrained=True):
        super().__init__()
        if model_name == 'resnet50':
            self.backbone = resnet50(pretrained=pretrained)
            self.backbone.fc = nn.Linear(in_features=2048, out_features=2)
        if model_name == 'mobilnet':
            self.backbone = mobilenet_v2(pretrained=pretrained)
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=False),
                nn.Linear(in_features=1280, out_features=2, bias=True)
            )

    def forward(self, x):
        x = self.backbone(x)
        x = F.softmax(x, dim=-1)
        return x


if __name__ == '__main__':
    model = FaceClassifier(model_name='mobilnet')
    print(model)
