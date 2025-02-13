from typing import Any

import numpy as np
from torch import Tensor

from .._criterion import Criterion
from ._prepare_tensor import prepare_tensor


class EuclideanDistance(Criterion):
    """Implements a Euclidean Distance measure."""

    _name: str = "EuclideanDistance"
    _normalize: bool

    def __init__(self, normalize: bool = False) -> None:
        """
        Initialize the EuclideanDistance measure.

        :param normalize: Whether to normalize the computed distance [0,1].
        """
        super().__init__()
        self._normalize = normalize

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

        return (
            np.linalg.norm(i1 - i2)
            if not self._normalize
            else np.linalg.norm(i1 - i2) / np.sqrt(len(i1))
        )
