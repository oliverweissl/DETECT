from typing import Any

import numpy as np
from numpy.typing import NDArray
from torch import Tensor

from .._criterion import Criterion
from ._prepare_tensor import prepare_tensor


class CFrobeniusDistance(Criterion):
    """Implements a channel-wise Frobenius Distance measure."""

    _name: str = "CFrobDistance"

    def evaluate(self, *, images: list[Tensor], **_: Any) -> float:
        """
        Calculate the normalized frobenius distance between two tensors that range [0,1].

        :param images: Images to compare.
        :param _: Additional unused kwargs.
        :returns: The distance.
        """
        assert len(images) == 2, f"ERROR, {self._name} requires 2 images, found {len(images)}"
        images = [prepare_tensor(i) for i in images]
        i1, i2 = images[0], images[1]

        ub = self._frob(np.ones(i1.shape[:-1]))
        mean_fn = (
            sum([self._frob(i1[..., j] - i2[..., j]) for j in range(i1.shape[-1])]) / i1.shape[-1]
        )
        return abs(self._inverse.real - mean_fn / ub)

    @staticmethod
    def _frob(matrix: NDArray) -> float:
        """
        Calculate the frobenius norm for a NxN matrix.

        :param matrix: The matrix to calculate with.
        :returns: The norm.
        """
        return np.sqrt(np.trace(matrix.T @ matrix))
