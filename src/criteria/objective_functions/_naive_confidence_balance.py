from typing import Any

from .._criterion import Criterion
from .._default_arguments import DefaultArguments


class NaiveConfidenceBalance(Criterion):
    """Implements a naive confidence balance measure."""

    def evaluate(self, *, default_args: DefaultArguments, **_: Any) -> float:
        """
        Calculate the confidence balance of two confidence values.

        This functions assumes input range of [0,1] and has a output range of [0,1].
        If y1 = 0, y2 = 1 -> 0/2 = 0.

        :param default_args: The default args parsed by the NeuralTester.
        :param _: Unused kwargs.
        :returns: The value.
        """
        return (default_args.y1p - default_args.y2p + 1) / 2
