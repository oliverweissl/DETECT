import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.ndimage import gaussian_filter
from sewar import msssim
from sewar import uqi as s_uqi


def w1_distance(i1: NDArray, i2: NDArray) -> float:
    """
    Get the wasserstein-1 distance from two distributions.

    :param i1: Distribution 1.
    :param i2: Distribution 2.
    :return: The distance [0,1].
    """
    raise NotImplementedError("This function is not yet implemented")
    m, n = len(i1), len(i2)

    i1 /= i1.sum()
    i2 /= i2.sum()

    A_u = sparse.block_diag((np.ones((1, n)),) * m)
    A_l = sparse.hstack((sparse.eye(n),) * m)

    A = sparse.vstack((A_u, A_l))
    A = sparse.coo_array(A)

    u1, u2 = i1.shape
    v1, v2 = i2.shape
    # TODO: this is WIP

    distance = 1  # wasserstein_distance_nd(u, v)
    assert 1 >= distance >= 0, "Distance does something weird."
    return distance


def ssim_d2(i1: NDArray, i2: NDArray) -> float:
    """
    Get structural similarity between two images as D_2 metric.

    Implementation based on https://github.com/scikit-image/scikit-image/blob/v0.24.0/skimage/metrics/_structural_similarity.py#L15-L292.
    And https://ece.uwaterloo.ca/~z70wang/publications/TIP_SSIM_MathProperties.pdf.
    Due to numpy conflicts with cuda we had to do our own implementation.

    :param i1: The first image (expects images of shape: CxHxW).
    :param i2: The second image (expects images of shape: CxHxW).
    :returns: SSIM score.
    """
    truncate, sigma = 3.5, 1.5
    assert (
        i1.shape == i2.shape
    ), f"Error: Both images need to be of same size ({i1.shape}, {i2.shape})."
    filter_curry = lambda image: gaussian_filter(image, sigma=sigma, truncate=truncate)
    pad = (2 * int(truncate * sigma + 0.5)) // 2

    ux, uy = filter_curry(i1), filter_curry(i2)  # local mean of x and y
    uxx, uyy, uxy = filter_curry(i1 * i1), filter_curry(i2 * i2), filter_curry(i1 * i2)

    vx = uxx - ux * ux  # local variance of x
    vy = uyy - uy * uy  # local variance of y
    vxy = uxy - ux * uy  # local covariance between x and y

    c1 = (0.01 * 1) ** 2.0  # (K1 * Data-Range)²
    c2 = (0.03 * 1) ** 2.0  # (K2 * Data-Range)²

    a1 = 2.0 * ux * uy + c1
    a2 = 2.0 * vxy + c2
    b1 = ux**2.0 + uy**2.0 + c1
    b2 = vx + vy + c2

    s1 = np.clip(a1 / b1, 0, 1)
    s2 = np.clip(a2 / b2, 0, 1)
    d = np.sqrt(2.0 - s1 - s2)

    d2 = d[:, pad:-pad, pad:-pad].mean()
    return d2 / np.sqrt(2)


def ms_ssim(i1: NDArray, i2: NDArray) -> float:
    """
    Get the Multi-Scale SSIM score.

    This score is in range (0,1) with 1 being the optimum.

    :param i1: The base image.
    :param i2: The modified image.
    :returns: The score.
    """
    i1 = i1.transpose(1, 2, 0)
    i2 = i2.transpose(1, 2, 0)
    return msssim(i1, i2, MAX=1.0).real


def uqi(i1: NDArray, i2: NDArray) -> float:
    """
    Get the Universal Image Quality Index score.

    This score is in range (0,1) with 1 being the optimum.

    :param i1: The base image.
    :param i2: The modified image.
    :returns: The score.
    """
    i1 = i1.transpose(1, 2, 0)
    i2 = i2.transpose(1, 2, 0)
    return s_uqi(i1, i2)
