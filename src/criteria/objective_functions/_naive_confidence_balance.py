from typing import Any, Optional

from torch import Tensor

from .._criterion import Criterion


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

    def evaluate(self, *, logits: Tensor, label_targets: list[int], **_: Any) -> float:
        """
        Calculate the confidence balance of two confidence values.

        This functions assumes input range of [0, 1].

        :param logits: The predicted logits.
        :param label_targets: The target labels in question.
        :param _: Unused kwargs.
        :returns: The value.
        """
        # TODO: investigate to improve this since |d| makes this non-linear.
        c1, c2 = label_targets[:2]
        y1p, y2p = logits[c1], logits[c2]

        s = y1p + y2p
        d = abs(y1p - y2p)

        if self._target_primary is None:
            return abs(self._inverse.real - d / s)
        else:
            return abs(self._inverse.real - (y2p if self._target_primary else y1p) - d / s)
