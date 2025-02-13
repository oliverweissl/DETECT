"""A collection of criteria used for comparing images (matrices)."""

from ._c_frobenius_distance import CFrobeniusDistance
from ._cos_dissimilarity import CosDissimilarity
from ._euclidean_distance import EuclideanDistance
from ._ms_ssim import MSSSIM
from ._ssim_d2 import SSIMD2
from ._uqi import UQI

__all__ = ["SSIMD2", "MSSSIM", "UQI", "CosDissimilarity", "CFrobeniusDistance", "EuclideanDistance"]
