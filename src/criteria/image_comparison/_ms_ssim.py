from typing import Any

from sewar import msssim

from .._criteria_arguments import CriteriaArguments
from .._criterion import Criterion
from ._prepare_tensor import prepare_tensor


class MSSSIM(Criterion):
    """Implements the Multi-Scale SSIM using sewar."""

    _name: str = "MS-SSIM"

    def evaluate(self, *, default_args: CriteriaArguments, **_: Any) -> float:
        """
        Get the Multi-Scale SSIM score.

        This score is in range (0,1) with 0 being the optimum.

        :param default_args: The default arguments parsed by the NeuralTester.
        :param _: Additional unused kwargs.
        :returns: The score.
        """
        i1, i2 = prepare_tensor(default_args.i1), prepare_tensor(default_args.i2)
        return 1 - msssim(i1, i2, MAX=1.0).real if self._inverse else msssim(i1, i2, MAX=1.0).real
