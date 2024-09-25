import numpy as np
from numpy.typing import NDArray
from scipy.stats import wasserstein_distance_nd

def get_w1_distance(u: NDArray, v: NDArray) -> float:
    """
    Get the wasserstein-1 distance from two distributions.

    :param u: Distribution 1.
    :param v: Distribution 2.
    :return: The distance [0,1].
    """
    u /= u.sum()
    v /= v.sum()

    distance = wasserstein_distance_nd(u, v)
    assert 1>= distance >= 0, "Distance does something weird."
    return distance
