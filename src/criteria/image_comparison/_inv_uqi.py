from typing import Any

from sewar import uqi
from torch import Tensor

from .._criterion import Criterion
from ._prepare_tensor import prepare_tensor


class InvUQI(Criterion):
    """Implements the inverse universal image quality index using sewar."""

    def evaluate(self, *, i1: Tensor, i2: Tensor, **_: Any) -> float:
        """
        Get the inversed Universal Image Quality Index score.

        This score is in range (0,1) with 0 being the optimum.

        :param i1: The base image.
        :param i2: The modified image.
        :param _: Additional unused kwargs.
        :returns: The score.
        """
        i1, i2 = prepare_tensor(i1), prepare_tensor(i2)
        return 1 - uqi(i1, i2)
