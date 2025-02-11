from typing import Any

from torch import Tensor

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

    def evaluate(self, *, logits: Tensor, **_: Any) -> float:
        """
        Calculate the accuracy from prediction probabilities.

        :param logits: Prediction probabilities.
        :param _: Unused kwargs.
        :return: The distance to the uncertainty threshold.
        """
        ypm = logits.max()
        dist = self._threshold - ypm.item()
        return abs(dist) if self._absolute else dist
