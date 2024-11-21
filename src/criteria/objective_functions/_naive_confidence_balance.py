from typing import Any

from .._criterion import Criterion
from .._default_arguments import DefaultArguments


class NaiveConfidenceBalance(Criterion):
    """Implements a naive confidence balance measure."""

    _name: str = "NaiveCB"

    def evaluate(self, *, default_args: DefaultArguments, **_: Any) -> float:
        """
        Calculate the confidence balance of two confidence values.

        This functions assumes input range of [0, 1].

        :param default_args: The default args parsed by the NeuralTester.
        :param _: Unused kwargs.
        :returns: The value.
        """
        # TODO: investigate to improve this since |d| makes this non-linear.
        s = default_args.y1p + default_args.y2p
        d = default_args.y1p - default_args.y2p
        return abs(d) / s
