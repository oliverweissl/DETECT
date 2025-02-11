from typing import Any

from sewar import msssim
from torch import Tensor

from .._criterion import Criterion
from ._prepare_tensor import prepare_tensor


class MSSSIM(Criterion):
    """Implements the Multi-Scale SSIM using sewar."""

    _name: str = "MS-SSIM"

    def evaluate(self, *, images: list[Tensor], **_: Any) -> float:
        """
        Get the Multi-Scale SSIM score.

        This score is in range (0,1) with 0 being the optimum.

        :param images: The images to compare.
        :param _: Additional unused kwargs.
        :returns: The score.
        """
        assert len(images) == 2, f"ERROR, {self._name} requires 2 images, found {len(images)}"
        images = [prepare_tensor(i) for i in images]
        i1, i2 = images[0], images[1]

        return 1 - msssim(i1, i2, MAX=1.0).real if self._inverse else msssim(i1, i2, MAX=1.0).real
