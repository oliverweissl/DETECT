from typing import Callable

from numpy.typing import NDArray
from torch import Tensor


def get_penalized_distance(
    img: Tensor,
    p_img: Tensor,
    y_img: int,
    y_p_img: int,
    metric: Callable[[NDArray, NDArray], float],
) -> float:
    """
    Get penalized distance between two images using their labels.

    :param img: The original image.
    :param p_img: The perturbed image.
    :param y_img: The original images label.
    :param y_p_img: The perturbed images label.
    :param metric: The metric to calculate.
    :return: The distance measure [0,1].
    """
    score = metric(img.cpu().numpy(), p_img.cpu().numpy())
    distance = (1 - score) ** (0 if y_p_img == y_img else 1)
    return distance
