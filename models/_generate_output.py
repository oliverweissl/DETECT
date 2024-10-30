import torch
import dnnlib
import legacy

import numpy as np
from torch import Tensor

def generate_output(
        model_pkl: str,
        seeds: list[int],
        class_index: int,
) -> list[Tensor]:
    """
    Generate datapoint from image.

    :param model_pkl: The destination of the model pickel file.
    :param seeds: A list of seeds for generation.
    :param class_index: The class index for class to be generated.
    :returns: One or multiple images of certain class.
    """
    """Set device and load generator."""
    device = torch.device("cuda")
    with dnnlib.util.open_url(model_pkl) as f:
        generator = legacy.load_network_pkl(f)["G_ema"].to(device)

    label = torch.zeros(size=[1, generator.c_dim], device=device)
    label[:, class_index] = 1


    z_generator = torch.Generator(device=device)
    images = []
    for idx, seed in enumerate(seeds):
        z_generator.manual_seed(seed)
        z = torch.randn(size=[1, generator.z_dim], device=device, generator=z_generator)

        if hasattr(generator.synthesis, 'input'):
            m = _make_transform((0.,0.), 0.)
            m = np.linalg.inv(m)
            generator.synthesis.input.transform.copy_(torch.from_numpy(m))

        seed = generator(z, label, noise_mode="random")
        image = (seed.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        images.append(image)
    return images


def _make_transform(translate: tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m


