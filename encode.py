from argparse import ArgumentParser

import numpy as np
import torch
import torchvision.transforms as tf
from PIL import Image
import pickle
from arithmetic_compressor import AECompressor
from arithmetic_compressor.models import SimpleAdaptiveModel

from autoencoder.encoder import ResNet18Encoder
from utils import QUANTIZE_MAP


def quantize(encoded, mode):
    return (encoded * (2 ** QUANTIZE_MAP[mode]) + 0.5).astype(int)


def arithm_encode(embedding, mode):
    keys = [key for key in range(0, 2 ** QUANTIZE_MAP[mode] + 1)]
    prob = 1. / len(keys)
    model = SimpleAdaptiveModel({k: prob for k in keys})
    coder = AECompressor(model)

    return coder.compress(embedding), len(embedding)


def encode():
    parser = ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--quantize_mode", type=str, required=True, choices=["hard", "soft"])
    parser.add_argument("--encoder_path", type=str, default="models/encoder.model")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = ResNet18Encoder().to(device)
    encoder.load_state_dict(torch.load(args.encoder_path, map_location=torch.device(device)))
    encoder.eval()

    img = Image.open(args.input_path)
    sample = tf.ToTensor()(img).unsqueeze(0).to(device)

    encoded = torch.clamp(encoder(sample).cpu(), 0.0, 1.0)[0].detach().numpy()
    quantized = quantize(encoded, args.quantize_mode)
    data, length = arithm_encode(quantized, args.quantize_mode)
    result = str(int("".join(map(str, [1] + data)), 2)) + "," + str(length)
    with open(args.output_path, 'wb') as f:
        pickle.dump(result, f)


if __name__ == '__main__':
    encode()
