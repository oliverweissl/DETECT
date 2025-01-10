from typing import Any, Optional

from .._criterion import Criterion
from .._criteria_arguments import CriteriaArguments


class NaiveConfidenceBalance(Criterion):
    """Implements a naive confidence balance measure."""

    _name: str = "NaiveCB"
    _target_primary: Optional[bool]

    def __init__(self, inverse: bool = False, target_primary: Optional[bool] = None) -> None:
        """
        Initialize the criterion.

        :param inverse: Whether the measure should be inverted.
        :param target_primary: Whether y1 is focus of the measure or y2, if none neither is in focus.
        """
        super().__init__(inverse=inverse)
        self._target_primary = target_primary

    def evaluate(self, *, default_args: CriteriaArguments, **_: Any) -> float:
        """
        Calculate the confidence balance of two confidence values.

        This functions assumes input range of [0, 1].

        :param default_args: The default args parsed by the NeuralTester.
        :param _: Unused kwargs.
        :returns: The value.
        """
        # TODO: investigate to improve this since |d| makes this non-linear.
        s = default_args.y1p + default_args.y2p
        d = abs(default_args.y1p - default_args.y2p)

        if self._target_primary is None:
            return abs(self._inverse.real - d / s)
        else:
            return abs(
                self._inverse.real
                - (default_args.y2p if self._target_primary else default_args.y1p)
                - d / s
            )
