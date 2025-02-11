from typing import Any

from torch import Tensor

from .._criterion import Criterion


class IsMisclassified(Criterion):
    """Implements a criterion to check if a prediction is incorrect."""

    _name: str = "IsMisclassified"

    def evaluate(self, *, logits: Tensor, label_targets: list[int], **_: Any) -> float:
        """
        Check if a prediction is incorrect.

        This functions assumes input range of [0, 1].

        :param logits: Tensor of predictions.
        :param label_targets: Label targets.
        :param _: Unused kwargs.
        :returns: The value.
        """
        c1 = label_targets[0]
        result = c1 == logits.argmax().item() if self._inverse else c1 != logits.argmax().item()
        return result
