import torch
from torch import nn
import torch.nn.functional as F

from autoencoder.decoder import ResNet18Decoder, ResNetDecBlock
from autoencoder.encoder import ResNet18Encoder


class ResNetAutoencoder(nn.Module):
    def __init__(self, quantize_factor):
        super().__init__()
        self.enc = ResNet18Encoder()
        self.dec = ResNet18Decoder(ResNetDecBlock)
        self.quantize_factor = quantize_factor

    def forward(self, x):
        out = self.enc(x)
        out = F.sigmoid(out)
        out = out + (1 / 2 ** self.quantize_factor) * (torch.rand_like(out) * 0.5 - 0.5)
        out = torch.log(out / (1 - out))
        out = self.dec(out)
        return F.sigmoid(out)