from typing import Any

import numpy as np
from torch import Tensor

from .._criterion import Criterion
from ._prepare_tensor import prepare_tensor


class CFrobeniusDistance(Criterion):
    """Implements a channel-wise Frobenius Distance measure."""

    def evaluate(self, *, i1: Tensor, i2: Tensor, **_: Any) -> float:
        """
        Calculate the normalized frobenius distance between two tensors [0,1].

        :param i1: The base image.
        :param i2: The modified image.
        :param _: Additional unused kwargs.
        :returns: The distance.
        """
        i1, i2 = prepare_tensor(i1), prepare_tensor(i2)
        ub = np.sqrt((255**2) * np.prod(i1.shape[:-1]))  # Distance supremum.
        return np.mean(np.linalg.norm(i1 - i2, ord="fro", axis=(0, 1))) / ub
