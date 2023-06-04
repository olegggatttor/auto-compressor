import pickle
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from arithmetic_compressor import AECompressor
from arithmetic_compressor.models import SimpleAdaptiveModel

from autoencoder.decoder import ResNet18Decoder, ResNetDecBlock
from utils import QUANTIZE_MAP


def dequantize(encoded, mode):
    return (encoded / (2 ** QUANTIZE_MAP[mode])).astype(float)


def arithm_decode(data, length, mode):
    keys = [key for key in range(0, 2 ** QUANTIZE_MAP[mode] + 1)]
    prob = 1. / len(keys)
    model = SimpleAdaptiveModel({k: prob for k in keys})
    coder = AECompressor(model)

    return coder.decompress(data, length)


def decode():
    parser = ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--quantize_mode", type=str, required=True, choices=["hard", "soft"])
    parser.add_argument("--decoder_path", type=str, default="models/decoder.model")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    decoder = ResNet18Decoder(ResNetDecBlock).to(device)
    decoder.load_state_dict(torch.load(args.decoder_path, map_location=torch.device(device)))
    decoder.eval()

    with open(args.input_path, 'rb') as f:
        encoded = pickle.load(f)

    [data, length] = encoded.split(",")
    bits = list(map(int, list(str(bin(int(data)))[3:])))  # drop 0b and one dummy bit
    to_decode = np.asarray(arithm_decode(bits, int(length), args.quantize_mode))

    encoded_tensor = torch.from_numpy(dequantize(to_decode, args.quantize_mode)).unsqueeze(0).float()
    decoded = F.sigmoid(decoder(encoded_tensor).cpu())[0].permute(1, 2, 0).detach().numpy()

    image = Image.fromarray((255 * decoded).astype(np.uint8))
    image.save(args.output_path)


if __name__ == '__main__':
    decode()
