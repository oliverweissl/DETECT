from typing import Any

from sewar import uqi
from torch import Tensor

from .._criterion import Criterion
from ._prepare_tensor import prepare_tensor


class UQI(Criterion):
    """Implements the universal image quality index using sewar."""

    _name: str = "UQI"

    def evaluate(self, *, images: list[Tensor], **_: Any) -> float:
        """
        Get the Universal Image Quality Index score.

        This score is in range (0,1) with 0 being the optimum.

        :param images: Images to compare.
        :param _: Additional unused kwargs.
        :returns: SSIM score.
        """
        assert len(images) == 2, f"ERROR, {self._name} requires 2 images, found {len(images)}"
        images = [prepare_tensor(i) for i in images]
        i1, i2 = images[0], images[1]

        return 1 - uqi(i1, i2) if self._inverse else uqi(i1, i2)
