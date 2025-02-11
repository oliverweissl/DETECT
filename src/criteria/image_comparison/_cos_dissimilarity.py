from typing import Any

import numpy as np
from torch import Tensor

from .._criterion import Criterion
from ._prepare_tensor import prepare_tensor


class CosDissimilarity(Criterion):
    """Implements cos dissimilarity measure."""

    _name: str = "CosDissim"

    def evaluate(self, *, images: list[Tensor], **_: Any) -> float:
        """
        Get the cosine-dissimilarity between two images.

        Range [0,1] with 0 being the same image.

        :param images: Images to compare.
        :param _: Additional unused kwargs.
        :returns: The score.
        """
        assert len(images) == 2, f"ERROR, {self._name} requires 2 images, found {len(images)}"
        images = [prepare_tensor(i) for i in images]
        i1, i2 = images[0], images[1]

        value = np.dot(i1.flatten(), i2.flatten()) / (np.linalg.norm(i1) * np.linalg.norm(i2))
        return 1 - value
