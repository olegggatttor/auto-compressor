import torch.nn as nn


class ResNetDecBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels))
        if stride != 1:
            self.conv2 = nn.Sequential(
                nn.Upsample(scale_factor=stride),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels))
            self.shortcut = nn.Sequential(
                nn.Upsample(scale_factor=stride),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        else:
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels))
            self.shortcut = nn.Sequential()
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet18Decoder(nn.Module):
    def __init__(self, Block):
        super().__init__()
        self.inchannel = 512
        self.up = nn.Upsample(scale_factor=2)
        self.layer1 = self.make_layer(Block, 256, 2, stride=2)
        self.layer2 = self.make_layer(Block, 128, 2, stride=2)
        self.layer3 = self.make_layer(Block, 64, 2, stride=2)
        self.layer4 = self.make_layer(Block, 64, 2, stride=1)
        self.resize = nn.Sequential(
            nn.Upsample(scale_factor=4),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1))

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = x.view(-1, 512, 8, 8)
        out = self.up(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.resize(out)
        return out
