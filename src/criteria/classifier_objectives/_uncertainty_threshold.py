from typing import Any
from .._criteria_arguments import CriteriaArguments

from .._criterion import Criterion


class UncertaintyThreshold(Criterion):
    """Implements uncertainty threshold as a criterion."""

    _name: str = "Uncertainty Threshold"
    _threshold: float
    _absolute: bool

    def __init__(self, threshold: float, absolute: bool = False) -> None:
        """
        Initialize uncertainty threshold.

        :param threshold: Threshold for the uncertainty.
        :param absolute: If true, compute the absolute distance to threshold.
        """
        super().__init__()
        self._threshold = threshold
        self._absolute = absolute

    def evaluate(self, *, default_args: CriteriaArguments, **_: Any) -> float:
        """
        Calculate the accuracy from prediction probabilities.

        :param default_args: Arguments passed to the current evaluation.
        :return: The distance to the uncertainty threshold.
        """
        ypm = default_args.yp.max()
        dist = self._threshold - ypm.item()
        return abs(dist) if self._absolute else dist
