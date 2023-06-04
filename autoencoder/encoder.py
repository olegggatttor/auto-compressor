import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet18_Weights


class ResNet18Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)

        self.conv1 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        return out