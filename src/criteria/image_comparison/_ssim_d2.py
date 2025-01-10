from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from .._criterion import Criterion
from .._criteria_arguments import CriteriaArguments
from ._prepare_tensor import prepare_tensor


class SSIMD2(Criterion):
    """
    Implements SSIM metric.

    Implementation based on https://github.com/scikit-image/scikit-image/blob/v0.24.0/skimage/metrics/_structural_similarity.py#L15-L292.
    And https://ece.uwaterloo.ca/~z70wang/publications/TIP_SSIM_MathProperties.pdf.
    Due to numpy conflicts with cuda we had to do our own implementation.
    """

    """Parameters for the evaluation."""
    truncate: float
    sigma: float
    k1: float
    k2: float

    def __init__(
        self, truncate: float = 3.5, sigma: float = 1.5, k1: float = 0.01, k2: float = 0.03
    ) -> None:
        """
        Initialize the SSIM D2 metric.

        :param truncate: Truncation value for the gaussian.
        :param sigma: Sigma value for the gaussian.
        :param k1: The K1 coefficient.
        :param k2: The K2 coefficient.
        """
        self.truncate, self.sigma, self.k1, self.k2 = truncate, sigma, k1, k2
        self._name = "SSIM-D2"

    def evaluate(
        self,
        *,
        default_args: CriteriaArguments,
        **_: Any,
    ) -> float:
        """
        Get structural similarity between two images as D_2 metric.

        :param default_args: The default arguments parsed by the NeuralTeser.
        :param _: Additional unused kwargs.
        :returns: SSIM score.
        """
        i1, i2 = prepare_tensor(default_args.i1), prepare_tensor(default_args.i2)
        assert (
            i1.shape == i2.shape
        ), f"Error: Both images need to be of same size ({i1.shape}, {i2.shape})."
        filter_curry = lambda image: gaussian_filter(
            image, sigma=self.sigma, truncate=self.truncate
        )
        pad = (2 * int(self.truncate * self.sigma + 0.5)) // 2

        ux, uy = filter_curry(i1), filter_curry(i2)  # local mean of x and y
        uxx, uyy, uxy = filter_curry(i1 * i1), filter_curry(i2 * i2), filter_curry(i1 * i2)

        vx = uxx - ux * ux  # local variance of x
        vy = uyy - uy * uy  # local variance of y
        vxy = uxy - ux * uy  # local covariance between x and y

        c1 = (self.k1 * 1) ** 2.0  # (K1 * Data-Range)²
        c2 = (self.k2 * 1) ** 2.0  # (K2 * Data-Range)²

        a1 = 2.0 * ux * uy + c1
        a2 = 2.0 * vxy + c2
        b1 = ux**2.0 + uy**2.0 + c1
        b2 = vx + vy + c2

        s1 = np.clip(a1 / b1, 0, 1)
        s2 = np.clip(a2 / b2, 0, 1)
        d = np.sqrt(2.0 - s1 - s2)

        d2 = d[pad:-pad, pad:-pad, :].mean()
        return d2 / np.sqrt(2)
