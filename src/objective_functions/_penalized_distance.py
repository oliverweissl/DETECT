import numpy as np
from torch import Tensor

from ._ssim import get_ssim_d2


def get_penalized_distance(
    img: Tensor, p_img: Tensor, y_img: int, y_p_img: int
) -> float:
    """
    Get penalized distance between two images using their labels.

    :param img: The original image.
    :param p_img: The perturbed image.
    :param y_img: The original images label.
    :param y_p_img: The perturbed images label.
    :return: The distance measure [0,1].
    """
    distance = (np.sqrt(2) - get_ssim_d2(img.cpu().numpy(), p_img.cpu().numpy())) ** (
        0 if y_p_img == y_img else 1
    )
    return distance
