"""
A collection of functions to compute image similarity from the sewar library.

Note that all functions have to be in range [0,1] to be useful.
"""

from numpy.typing import NDArray
from sewar.full_ref import msssim
from sewar.full_ref import uqi as s_uqi


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
