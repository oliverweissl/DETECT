from typing import Any, Callable

from numpy.typing import NDArray
from torch import Tensor


def penalized_distance(
    *,
    i1: Tensor,
    i2: Tensor,
    y1: int,
    y2: int,
    metric: Callable[[NDArray, NDArray], float],
) -> float:
    """
    Get penalized distance between two images using their labels.

    :param i1: The original image.
    :param i2: The perturbed image.
    :param y1: The original images label.
    :param y2: The perturbed images label.
    :param metric: The metric to calculate.
    :return: The distance measure [0,1].
    """
    score = metric(i1.cpu().numpy(), i2.cpu().numpy())
    distance = (1 - score) ** (0 if y2 < y1 else 1)
    return distance


def confidence_balance(*, y1: float, y2: float, **_: Any) -> float:
    """
    Calculate the confidence balence of two cofidence values.

    :param y1: The primary confidence.
    :param y2: The secondary confidence.
    :param _: Unused kwargs.
    :returns: The value.
    """
    return (y2 - y1 + 1) / 2
