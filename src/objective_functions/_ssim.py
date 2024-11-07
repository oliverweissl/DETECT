import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter


def get_ssim_d2(i1: NDArray, i2: NDArray) -> float:
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
    return d2
