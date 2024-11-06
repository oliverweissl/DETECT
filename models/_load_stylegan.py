import torch

import dnnlib
import legacy


def load_stylegan(file: str) -> torch.nn.Module:
    """
    Load a StyleGAN network from pkl file.

    :param file: The file path.
    :returns: The module.
    """
    with dnnlib.util.open_url(file) as f:
        return legacy.load_network_pkl(f)["G_ema"]
