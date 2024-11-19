from typing import Any

import numpy as np
from torch import Tensor

from .._criterion import Criterion
from ._prepare_tensor import prepare_tensor


class CosDissimilarity(Criterion):
    """Implements cos dissimilarity measure."""

    def evaluate(self, *, i1: Tensor, i2: Tensor, **_: Any) -> float:
        """
        Get the cosine-dissimilarity between two images.

        Range [0,1] with 0 being the same image.

        :param i1: The base image.
        :param i2: The modified image.
        :param _: Additional unused kwargs.
        :returns: The score.
        """
        i1, i2 = prepare_tensor(i1), prepare_tensor(i2)
        value = np.dot(i1.flatten(), i2.flatten()) / (np.linalg.norm(i1) * np.linalg.norm(i2))
        return 1 - value
