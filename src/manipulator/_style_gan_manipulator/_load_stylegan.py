import torch

from .dnnlib.util import open_url
from .legacy import load_network_pkl


def load_stylegan(file: str) -> torch.nn.Module:
    """
    Load a StyleGAN network from pkl file.

    :param file: The file path.
    :returns: The module.
    """
    with open_url(file) as f:
        return load_network_pkl(f)["G_ema"]
