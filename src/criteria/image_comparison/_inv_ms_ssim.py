from typing import Any

from sewar import msssim

from .._criterion import Criterion
from .._default_arguments import DefaultArguments
from ._prepare_tensor import prepare_tensor


class InvMSSSIM(Criterion):
    """Implements the inversed Multi-Scale SSIM using sewar."""

    def evaluate(self, *, default_args: DefaultArguments, **_: Any) -> float:
        """
        Get the inversed Multi-Scale SSIM score.

        This score is in range (0,1) with 0 being the optimum.

        :param default_args: The default arguments parsed by the NeuralTester.
        :param _: Additional unused kwargs.
        :returns: The score.
        """
        i1, i2 = prepare_tensor(default_args.i1), prepare_tensor(default_args.i2)
        return 1 - msssim(i1, i2, MAX=1.0).real
