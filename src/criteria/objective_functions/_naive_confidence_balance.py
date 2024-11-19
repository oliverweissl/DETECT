from typing import Any

from .._criterion import Criterion


class NaiveConfidenceBalance(Criterion):
    """Implements a naive confidence balance measure."""

    def evaluate(self, *, y1: float, y2: float, **_: Any) -> float:
        """
        Calculate the confidence balance of two confidence values.

        This functions assumes input range of [0,1] and has a output range of [0,1].
        If y1 = 0, y2 = 1 -> 0/2 = 0.

        :param y1: The primary confidence.
        :param y2: The secondary confidence.
        :param _: Unused kwargs.
        :returns: The value.
        """
        return (y1 - y2 + 1) / 2
