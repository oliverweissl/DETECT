"""A collection of criteria used for comparing images (matrices)."""

from ._c_frobenius_distance import CFrobeniusDistance
from ._cos_dissimilarity import CosDissimilarity
from ._inv_ms_ssim import InvMSSSIM
from ._inv_uqi import InvUQI
from ._ssim_d2 import SSIMD2

__all__ = ["SSIMD2", "InvMSSSIM", "InvUQI", "CosDissimilarity", "CFrobeniusDistance"]
